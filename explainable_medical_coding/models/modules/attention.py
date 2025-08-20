from typing import Optional, Callable
import random
import string

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from rich.progress import track
from explainable_medical_coding.utils.data_helper_functions import get_code2description_mimiciv_combined


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        output_attention: bool = False,
        attention_masks: Optional[torch.Tensor] = None,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, proj_size]
        att_weights = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x  # [batch_size, num_classes, input_size]
        output = (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )  # [batch_size, num_classes]
        if output_attention:
            return output, att_weights
        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)

class LabelCrossAttention(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_classes: int, 
        scale: float = 1.0,
        init_with_descriptions: bool = False,
        model_path: str = None,
        target_tokenizer = None,
        icd_version: int = 10,
        freeze_label_embeddings: bool = False,
        random_init: bool = False
    ):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = torch.nn.Parameter(
            torch.rand(num_classes, input_size), requires_grad=True
        )
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        
        # Initialise weights
        if init_with_descriptions and model_path and target_tokenizer is not None:
            if random_init:
                self._init_weights_random_descriptions(model_path, target_tokenizer, icd_version)
            else:
                self._init_weights_description_embeddings(model_path, target_tokenizer, icd_version)
        else:
            self._init_weights(mean=0.0, std=0.03)
        
        # Freeze label embeddings if requested
        if freeze_label_embeddings:
            self.freeze_label_embeddings()
        

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.label_representations

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)
            # repeat attention masks for each class
            attention_masks = attention_masks.unsqueeze(1).repeat(
                1, self.num_classes, 1
            )
            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]
        if attn_grad_hook_fn is not None:
            attention.register_hook(attn_grad_hook_fn)

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(
            self.label_representations, mean, std
        )
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

    def _init_weights_description_embeddings(
        self, 
        model_path: str, 
        target_tokenizer,
        icd_version: int = 10,
        batch_size: int = 64,
        mean: float = 0.0, 
        std: float = 0.03
    ) -> None:
        """Initialize label representations with ICD code description embeddings.

        Args:
            model_path (str): Path to the transformer model for generating embeddings
            target_tokenizer: TargetTokenizer that maps indices to ICD codes
            icd_version (int): ICD version (9 or 10). Defaults to 10.
            batch_size (int): Batch size for processing descriptions. Defaults to 64.
        """
        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

        # Load ICD descriptions
        code2desc = get_code2description_mimiciv_combined(icd_version)
        
        # Load embedding model
        embed_model = AutoModel.from_pretrained(model_path)
        embed_model = embed_model.cuda()
        embed_tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Collect all descriptions in target_tokenizer order
        descriptions = []
        for i in range(len(target_tokenizer)):
            icd_code = target_tokenizer.id2target[i]
            descriptions.append(code2desc[icd_code])
        
        # Process descriptions in batches with progress bar
        all_embeddings = []
        batch_ranges = range(0, len(descriptions), batch_size)
        
        for i in track(batch_ranges, description="Generating description embeddings..."):
            batch_descriptions = descriptions[i:i+batch_size]
            
            # Batch tokenization
            tokens = embed_tokenizer(
                batch_descriptions,
                return_tensors="pt", 
                truncation=True, 
                max_length=64,
                padding=True
            ).to('cuda')
            
            # Batch forward pass
            with torch.no_grad():
                outputs = embed_model(**tokens)
                # Mean pooling with attention mask to handle padding
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                masked_embeddings = outputs.last_hidden_state * attention_mask
                embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
                all_embeddings.append(embeddings)
        
        # Combine all embeddings and replace parameter data
        final_embeddings = torch.cat(all_embeddings, dim=0)
        self.label_representations.data.copy_(final_embeddings)
        
        # Clean up the temporary model to free memory
        del embed_model
        del embed_tokenizer

    def _init_weights_random_descriptions(
        self, 
        model_path: str, 
        target_tokenizer,
        icd_version: int = 10,
        batch_size: int = 64,
        max_desc_len: int = 64,
        mean: float = 0.0, 
        std: float = 0.03
    ) -> None:
        """Initialize label representations with embeddings from random text descriptions.

        Args:
            model_path (str): Path to the transformer model for generating embeddings
            target_tokenizer: TargetTokenizer that maps indices to ICD codes
            icd_version (int): ICD version (9 or 10). Defaults to 10.
            batch_size (int): Batch size for processing descriptions. Defaults to 64.
            max_desc_len (int): Maximum description length for tokenization. Defaults to 64.
        """
        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

        def generate_random_text(seed):
            """Generate random text with characters and spaces using class-specific seed."""
            rng = random.Random(42 + seed)
            result = []
            remaining_length = rng.randint(30, 130)  # Total character budget
            
            while remaining_length > 0:
                # Add random characters (3-10 chars)
                char_length = min(rng.randint(3, 10), remaining_length)
                chars = ''.join(rng.choices(string.ascii_letters + string.digits, k=char_length))
                result.append(chars)
                remaining_length -= char_length
                
                # Add space if there's still length remaining
                if remaining_length > 0:
                    result.append(' ')
                    remaining_length -= 1
            
            return ''.join(result)
        
        # Load embedding model
        embed_model = AutoModel.from_pretrained(model_path)
        embed_model = embed_model.cuda()
        embed_tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Generate random text for each target with different seeds
        descriptions = []
        for i in range(len(target_tokenizer)):
            desc = generate_random_text(i)
            descriptions.append(desc)
        
        # Process descriptions in batches with progress bar
        all_embeddings = []
        batch_ranges = range(0, len(descriptions), batch_size)
        
        for i in track(batch_ranges, description="Generating random description embeddings..."):
            batch_descriptions = descriptions[i:i+batch_size]
            
            # Batch tokenization
            tokens = embed_tokenizer(
                batch_descriptions,
                return_tensors="pt", 
                truncation=True, 
                max_length=max_desc_len,
                padding=True
            ).to('cuda')
            
            # Batch forward pass
            with torch.no_grad():
                outputs = embed_model(**tokens)
                # Mean pooling with attention mask to handle padding
                attention_mask = tokens['attention_mask'].unsqueeze(-1)
                masked_embeddings = outputs.last_hidden_state * attention_mask
                embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
                all_embeddings.append(embeddings)
        
        # Combine all embeddings and replace parameter data
        final_embeddings = torch.cat(all_embeddings, dim=0)
        self.label_representations.data.copy_(final_embeddings)
        
        # Clean up the temporary model to free memory
        del embed_model
        del embed_tokenizer

    def freeze_label_embeddings(self) -> None:
        """Freeze the label representation parameters."""
        self.label_representations.requires_grad = False
        
    def unfreeze_label_embeddings(self) -> None:
        """Unfreeze the label representation parameters."""
        self.label_representations.requires_grad = True

class LabelCrossAttentionDE(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        num_classes: int, 
        scale: float = 1.0,
        encoder_model = None,
        encoder_tokenizer = None,
        target_tokenizer = None,
        icd_version: int = 10,
        desc_batch_size: int = 64,
        random_init: bool = False,
    ):
        super().__init__()

        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        
        # Store encoder model and tokenizer
        self.encoder_model = encoder_model
        self.encoder_tokenizer = encoder_tokenizer
        self.desc_batch_size = desc_batch_size
        
        # Initialize tokenized descriptions as buffers
        if target_tokenizer is not None:
            if not random_init:
                self._init_description_tokens(target_tokenizer, icd_version)
            else:
                self._init_description_tokens_random(
                    encoder_tokenizer, target_tokenizer, icd_version, max_desc_len=64)
        
        # Initialize weights
        self._init_weights(mean=0.0, std=0.03)
    
    def _init_description_tokens(self, target_tokenizer, icd_version: int) -> None:
        """Initialize tokenized descriptions as buffers."""
        code2desc = get_code2description_mimiciv_combined(icd_version)
        descriptions = [code2desc[target_tokenizer.id2target[i]] for i in range(len(target_tokenizer))]
        
        tokens = self.encoder_tokenizer(
            descriptions,
            return_tensors="pt", 
            truncation=True, 
            max_length=64,
            padding=True
        )
        print(f"First token ID: {tokens.input_ids[0, 0]}")
        print(f"CLS token ID: {self.encoder_tokenizer.cls_token_id}")
        print(f"Are they equal? {tokens.input_ids[0, 0] == self.encoder_tokenizer.cls_token_id}")
        
        self.register_buffer("description_input_ids", tokens.input_ids)
        self.register_buffer("description_attention_mask", tokens.attention_mask)
    
    def _init_description_tokens_random(self, encoder_tokenizer, target_tokenizer, icd_version: int, max_desc_len: int) -> None:
        """Initialize tokenized descriptions with random text as buffers."""
        import random
        import string
        
        # First get real descriptions and tokenize them to get the attention masks
        code2desc = get_code2description_mimiciv_combined(icd_version)
        real_descriptions = [code2desc[target_tokenizer.id2target[i]] for i in range(len(target_tokenizer))]
        
        # Tokenize real descriptions to get their attention masks
        real_tokens = encoder_tokenizer(
            real_descriptions,
            return_tensors="pt", 
            truncation=True, 
            max_length=max_desc_len,
            padding=True
        )
        real_attention_mask = real_tokens.attention_mask
        
        def generate_random_text():
            """Generate 300 random characters with spaces."""
            result = []
            remaining_length = 300
            
            while remaining_length > 0:
                # Add random characters (3-10 chars)
                char_length = min(random.randint(3, 10), remaining_length)
                chars = ''.join(random.choices(string.ascii_letters + string.digits, k=char_length))
                result.append(chars)
                remaining_length -= char_length
                
                # Add space if there's still length remaining
                if remaining_length > 0:
                    result.append(' ')
                    remaining_length -= 1
            
            return ''.join(result)
        
        # Generate random descriptions for all classes
        descriptions = []
        random.seed(42)  # For reproducibility
        for i in range(len(target_tokenizer)):
            random_desc = generate_random_text()
            descriptions.append(random_desc)
        
        # Tokenize random descriptions (they'll be truncated to max_desc_len)
        random_tokens = encoder_tokenizer(
            descriptions,
            return_tensors="pt", 
            truncation=True, 
            max_length=max_desc_len,
            padding=True
        )
        
        # Replace random tokens with real attention patterns to ensure identical computation
        # Copy random token content but match the structure of real tokens exactly
        final_input_ids = real_tokens.input_ids.clone()
        final_attention_mask = real_tokens.attention_mask.clone()
        
        # For each description, replace the actual content tokens (non-padding) with random tokens
        for i in range(len(target_tokenizer)):
            # Find how many non-padding tokens the real description has
            real_length = real_tokens.attention_mask[i].sum().item()
            
            # Get random tokens for this description (excluding special tokens)
            random_content = random_tokens.input_ids[i]
            # Skip CLS token, take content tokens, avoid SEP/PAD tokens
            if encoder_tokenizer.cls_token_id is not None:
                content_start = 1
            else:
                content_start = 0
                
            # Replace real content with random content, keeping same structure
            content_length = real_length - content_start  # Account for CLS
            if encoder_tokenizer.sep_token_id is not None:
                content_length -= 1  # Account for SEP
                
            if content_length > 0:
                random_content_tokens = random_content[content_start:content_start + content_length]
                final_input_ids[i, content_start:content_start + len(random_content_tokens)] = random_content_tokens
        
        self.register_buffer("description_input_ids", final_input_ids)
        self.register_buffer("description_attention_mask", final_attention_mask)


    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
        encoded_descriptions= None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism with no projections for better alignment

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """


        V = self.weights_v(x)  # Projected clinical text embeddings
        K = self.weights_k(x)  # Projected clinical text embeddings
        
        Q = encoded_descriptions  # Raw code description embeddings from same encoder

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)
            # repeat attention masks for each class
            attention_masks = attention_masks.unsqueeze(1).repeat(
                1, self.num_classes, 1
            )
            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]
        if attn_grad_hook_fn is not None:
            attention.register_hook(attn_grad_hook_fn)

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """
        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

class TokenLevelDescriptionCrossAttention(nn.Module):
    """
    Performs cross-attention where each code description (Query) attends to the
    tokens of the clinical note (Key, Value).

    Not Used in the dissertation due to high memory requirements and computation time.
    dynamic_token_level_cross_attention.py is used instead.

    """
    def __init__(
        self, 
        input_size: int, 
        num_classes: int, 
        model_path: str,
        target_tokenizer,
        scale: float = 1.0,
        icd_version: int = 10,
        max_desc_len: int = 64
    ):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.scale = scale / (input_size ** 0.5)

        self.q_proj = nn.Linear(input_size, input_size, bias=False)
        self.k_proj = nn.Linear(input_size, input_size, bias=False)
        self.v_proj = nn.Linear(input_size, input_size, bias=False)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)

        embeddings_placeholder = torch.randn(num_classes, max_desc_len, input_size)
        self.register_buffer("description_embeddings", embeddings_placeholder)
        
        # Load and store the embedding model and tokenizer
        print("Loading description embedding model...")
        self.embed_model = AutoModel.from_pretrained(model_path)
        self.embed_tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Tokenize and store all code descriptions
        self._init_weights(target_tokenizer, icd_version, max_desc_len)

    def _init_weights(
        self, 
        target_tokenizer, 
        icd_version, 
        max_desc_len, 
        mean: float = 0.0, 
        std: float = 0.03
    ) -> None:
        
        self.k_proj.weight = torch.nn.init.normal_(self.k_proj.weight, mean, std)
        self.v_proj.weight = torch.nn.init.normal_(self.v_proj.weight, mean, std)
        self.q_proj.weight = torch.nn.init.normal_(self.q_proj.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

        code2desc = get_code2description_mimiciv_combined(icd_version)
        descriptions = [code2desc[target_tokenizer.id2target[i]] for i in range(len(target_tokenizer))]
        
        tokens = self.embed_tokenizer(
            descriptions, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=max_desc_len
        )
        
        print("Pre-computing description embeddings...")
        self.embed_model = self.embed_model.cuda()  
        
        with torch.no_grad():
            outputs = self.embed_model(
                input_ids=tokens.input_ids.cuda(),
                attention_mask=tokens.attention_mask.cuda()
            )
            self.description_embeddings = outputs.last_hidden_state
        
        self.register_buffer("description_attention_mask", tokens.attention_mask)

        del self.embed_model
        del self.embed_tokenizer
        
        print(f"Stored pre-computed embeddings of shape: {self.description_embeddings.shape}")


    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass where descriptions are the Query.

        Args:
            x (torch.Tensor): Clinical note embeddings. 
                              Shape: [batch_size, note_seq_len, input_size]
            attention_masks (torch.Tensor): Mask for the clinical note. 
                                            Shape: [batch_size, note_seq_len]
        Returns:
            torch.Tensor: Logits for each class. Shape: [batch_size, num_classes]
        """
        batch_size, note_seq_len, _ = x.shape
        
        q_desc = self.q_proj(self.description_embeddings)       # -> [num_classes, desc_seq, dim]
        k_note = self.k_proj(x)                 # -> [batch, note_seq, dim]
        v_note = self.v_proj(x)                 # -> [batch, note_seq, dim]

        # 'c,d,b,n' = class, desc_seq, batch, note_seq
        att_weights = torch.einsum('cdi,bni->cdbn', q_desc, k_note) * self.scale
        # -> att_weights shape: [num_classes, desc_seq, batch_size, note_seq]

        # Handle attention masks
        desc_mask = self.description_attention_mask.view(self.num_classes, -1, 1, 1)
        
        if attention_masks is not None:
            note_mask = attention_masks.view(1, 1, batch_size, note_seq_len)
            combined_mask = (desc_mask * note_mask).bool()
        else:
            # If no note mask provided, only use description mask
            combined_mask = desc_mask.bool()
            combined_mask = combined_mask.expand(-1, -1, batch_size, note_seq_len)
        
        att_weights = att_weights.permute(2, 0, 1, 3)
        combined_mask = combined_mask.permute(2, 0, 1, 3)
        att_weights.masked_fill_(~combined_mask, -1e9)

        attention = torch.softmax(att_weights, dim=-1)
        
        # 'b,c,d,n' @ 'b,n,i' -> 'b,c,d,i'
        context = torch.einsum('bcdn,bni->bcdi', attention, v_note)
        # -> context shape: [batch, num_classes, desc_seq_len, input_size]
        
        desc_mask_for_pooling = self.description_attention_mask.view(1, self.num_classes, -1, 1)
        context.masked_fill_(~desc_mask_for_pooling.bool(), 0)

        pooled_context = context.sum(dim=2) / self.description_attention_mask.sum(dim=1).view(1, self.num_classes, 1)
        # -> pooled_context shape: [batch, num_classes, input_size]

        y = self.layernorm(pooled_context)
        output = self.output_linear(y).squeeze(-1) # -> [batch, num_classes]
        
        return output

class InputMasker(nn.Module):
    def __init__(self, input_size: int, scale: float = 1.0):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size)
        self.weights_q = nn.Linear(input_size, input_size)
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.scale = scale
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.weights_q(x)

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        if attention_masks is not None:
            # pad attention masks with 0 such that it has the same sequence lenght as x
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, x.size(1) - attention_masks.size(1)), value=0
            )
            attention_masks = attention_masks.to(torch.bool)

            # repeat attention masks for each token
            attention_masks = attention_masks.unsqueeze(1).repeat(1, x.size(1), 1)

            attention_masks = attention_masks.masked_fill_(
                attention_masks.logical_not(), float("-inf")
            )
            att_weights += attention_masks

        attention = torch.softmax(
            att_weights / self.scale, dim=2
        )  # [batch_size, num_classes, seq_len]

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.weights_q.weight = torch.nn.init.normal_(self.weights_q.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )


class DynamicTokenLevelCrossAttention(nn.Module):
    """
    Token-level cross attention that works with dynamically provided encoded descriptions.
    Suitable for top-k retrieval where we only want to compute attention for a subset of codes.
    Handles tokenization and storage of all descriptions.
    """
    def __init__(
        self, 
        input_size: int, 
        scale: float = 1.0,
        encoder_tokenizer=None,
        target_tokenizer=None,
        icd_version: int = 10,
        max_desc_len: int = 64,
        pooling_temperature: float = 1.0,
        random_init: bool = False
    ):
        super().__init__()
        self.input_size = input_size
        self.scale = scale / (input_size ** 0.5)
        self.pooling_temperature = pooling_temperature

        # Add back projections
        self.q_proj = nn.Linear(input_size, input_size, bias=False)
        self.k_proj = nn.Linear(input_size, input_size, bias=False)
        self.v_proj = nn.Linear(input_size, input_size, bias=False)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        
        # Tokenize and store all descriptions
        if target_tokenizer is not None:
            if not random_init:
                self._init_description_tokens(encoder_tokenizer=encoder_tokenizer, target_tokenizer=target_tokenizer, icd_version=icd_version, max_desc_len=max_desc_len)
            else:
                self._init_description_tokens_random(
                    encoder_tokenizer=encoder_tokenizer, target_tokenizer=target_tokenizer, icd_version=icd_version, max_desc_len=max_desc_len)
                        
        self._init_weights()

    def _init_description_tokens(self, encoder_tokenizer, target_tokenizer, icd_version: int = 10, max_desc_len: int = 64) -> None:
        """Initialize tokenized descriptions as buffers."""
        code2desc = get_code2description_mimiciv_combined(icd_version)
        descriptions = [code2desc[target_tokenizer.id2target[i]] for i in range(len(target_tokenizer))]
        
        tokens = encoder_tokenizer(
            descriptions,
            return_tensors="pt", 
            truncation=True, 
            max_length=max_desc_len,
            padding=True
        )
        
        self.register_buffer("description_input_ids", tokens.input_ids)
        self.register_buffer("description_attention_mask", tokens.attention_mask)

    def _init_description_tokens_random(self, encoder_tokenizer, target_tokenizer, icd_version: int = 10, max_desc_len: int = 64) -> None:
        """Initialize tokenized descriptions with random text as buffers."""
        import random
        import string
        
        # First get real descriptions and tokenize them to get the attention masks
        code2desc = get_code2description_mimiciv_combined(icd_version)
        real_descriptions = [code2desc[target_tokenizer.id2target[i]] for i in range(len(target_tokenizer))]
        
        # Tokenize real descriptions to get their attention masks
        real_tokens = encoder_tokenizer(
            real_descriptions,
            return_tensors="pt", 
            truncation=True, 
            max_length=max_desc_len,
            padding=True
        )
        real_attention_mask = real_tokens.attention_mask
        
        def generate_random_text():
            """Generate 300 random characters with spaces."""
            result = []
            remaining_length = 300
            
            while remaining_length > 0:
                # Add random characters (3-10 chars)
                char_length = min(random.randint(3, 10), remaining_length)
                chars = ''.join(random.choices(string.ascii_letters + string.digits, k=char_length))
                result.append(chars)
                remaining_length -= char_length
                
                # Add space if there's still length remaining
                if remaining_length > 0:
                    result.append(' ')
                    remaining_length -= 1
            
            return ''.join(result)
        
        # Generate random descriptions for all classes
        descriptions = []
        random.seed(42)  # For reproducibility
        for i in range(len(target_tokenizer)):
            random_desc = generate_random_text()
            descriptions.append(random_desc)
        
        # Tokenize random descriptions (they'll be truncated to max_desc_len)
        random_tokens = encoder_tokenizer(
            descriptions,
            return_tensors="pt", 
            truncation=True, 
            max_length=max_desc_len,
            padding=True
        )
        
        # Replace random tokens with real attention patterns to ensure identical computation
        # Copy random token content but match the structure of real tokens exactly
        final_input_ids = real_tokens.input_ids.clone()
        final_attention_mask = real_tokens.attention_mask.clone()
        
        # For each description, replace the actual content tokens (non-padding) with random tokens
        for i in range(len(target_tokenizer)):
            # Find how many non-padding tokens the real description has
            real_length = real_tokens.attention_mask[i].sum().item()
            
            # Get random tokens for this description (excluding special tokens)
            random_content = random_tokens.input_ids[i]
            # Skip CLS token, take content tokens, avoid SEP/PAD tokens
            if encoder_tokenizer.cls_token_id is not None:
                content_start = 1
            else:
                content_start = 0
                
            # Replace real content with random content, keeping same structure
            content_length = real_length - content_start  # Account for CLS
            if encoder_tokenizer.sep_token_id is not None:
                content_length -= 1  # Account for SEP
                
            if content_length > 0:
                random_content_tokens = random_content[content_start:content_start + content_length]
                final_input_ids[i, content_start:content_start + len(random_content_tokens)] = random_content_tokens
        
        self.register_buffer("description_input_ids", final_input_ids)
        self.register_buffer("description_attention_mask", final_attention_mask)

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        self.q_proj.weight = torch.nn.init.normal_(self.q_proj.weight, mean, std)
        self.k_proj.weight = torch.nn.init.normal_(self.k_proj.weight, mean, std)
        self.v_proj.weight = torch.nn.init.normal_(self.v_proj.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

    def forward(
        self,
        x: torch.Tensor,
        encoded_descriptions: torch.Tensor,
        description_attention_mask: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with dynamically provided encoded descriptions.

        Args:
            x (torch.Tensor): Clinical note embeddings. 
                              Shape: [batch_size, note_seq_len, input_size]
            encoded_descriptions (torch.Tensor): Pre-encoded description embeddings.
                                               Shape: [batch_size, k, desc_seq_len, input_size]
            description_attention_mask (torch.Tensor): Mask for descriptions.
                                                      Shape: [batch_size, k, desc_seq_len]
            attention_masks (torch.Tensor): Mask for the clinical note. 
                                            Shape: [batch_size, note_seq_len]
        Returns:
            torch.Tensor: Logits for each of the k classes. Shape: [batch_size, k]
        """
        batch_size, note_seq_len, _ = x.shape
        _, k, desc_seq_len, _ = encoded_descriptions.shape
        
        # Project descriptions and notes
        q_desc = self.q_proj(encoded_descriptions)   # -> [batch_size, k, desc_seq, dim]
        k_note = self.k_proj(x)                      # -> [batch, note_seq, dim]
        v_note = self.v_proj(x)                      # -> [batch, note_seq, dim]

        # Cross attention: descriptions attend to note tokens
        # 'b,k,d,i' @ 'b,n,i' -> 'b,k,d,n'
        att_weights = torch.einsum('bkdi,bni->bkdn', q_desc, k_note) * self.scale
        # -> att_weights shape: [batch_size, k, desc_seq, note_seq]

        # Handle attention masks
        desc_mask = description_attention_mask.view(batch_size, k, desc_seq_len, 1)
        
        if attention_masks is not None:
            # Pad attention masks to match note sequence length
            attention_masks = torch.nn.functional.pad(
                attention_masks, (0, note_seq_len - attention_masks.size(1)), value=0
            )
            note_mask = attention_masks.view(batch_size, 1, 1, note_seq_len)
            combined_mask = (desc_mask * note_mask).bool()
        else:
            # If no note mask provided, only use description mask
            combined_mask = desc_mask.bool()
            combined_mask = combined_mask.expand(-1, -1, -1, note_seq_len)
        
        # att_weights is already [batch, k, desc_seq, note_seq]
        att_weights.masked_fill_(~combined_mask, -1e9)

        attention = torch.softmax(att_weights, dim=-1)
        
        # Apply attention to note values
        # 'b,k,d,n' @ 'b,n,i' -> 'b,k,d,i'
        context = torch.einsum('bkdn,bni->bkdi', attention, v_note)
        # -> context shape: [batch, k, desc_seq_len, input_size]
        
        # Mean pooling over description sequence dimension
        # Apply description mask to context before pooling
        desc_mask_for_pooling = description_attention_mask.view(batch_size, k, desc_seq_len, 1).bool()
        context.masked_fill_(~desc_mask_for_pooling, 0)
        
        # Mean pooling
        pooled_context = context.sum(dim=2) / description_attention_mask.sum(dim=2).view(batch_size, k, 1)
        # -> pooled_context shape: [batch, k, input_size]

        y = self.layernorm(pooled_context)
        output = (
            self.output_linear.weight.mul(y)                # [C, D] * [B, C, D]
            .sum(dim=2)                                     # dot product per class
            .add(self.output_linear.bias)                   # add class-specific bias
        )   
        
        return output
