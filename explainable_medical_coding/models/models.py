from typing import Callable, Optional

import torch
import torch.utils.checkpoint
from pydantic import BaseModel
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers import AutoTokenizer

from explainable_medical_coding.explainability.helper_functions import (
    create_baseline_input,
)
from explainable_medical_coding.models.modules.attention import (
    InputMasker,
    LabelAttention,
    LabelCrossAttention,
    LabelCrossAttentionDE,
    TokenLevelDescriptionCrossAttention,
    DynamicTokenLevelCrossAttention,
)


class ModuleNames(BaseModel):
    ln_1: str
    ln_2: str
    dense_values: str
    dense_heads: str
    model_layer_name: str


class RobertaModuleNames(ModuleNames):
    ln_1: str = "attention.output.LayerNorm"
    ln_2: str = "output.LayerNorm"
    dense_values: str = "attention.self.value"
    dense_heads: str = "attention.output.dense"
    model_layer_name: str = "roberta_encoder.encoder.layer"


class PLMICD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_path: str,
        chunk_size: int,
        pad_token_id: int,
        cross_attention: bool = True,
        attention_type: str = "label",  # "label", "token_level", "dual_encoding"
        random_init: bool = False,
        init_with_descriptions: bool = False,
        freeze_label_embeddings: bool = False,
        scale: float = 1.0,
        mask_input: bool = False,
        target_tokenizer = None,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.module_names = RobertaModuleNames()
        self.gradient = None
        self.attention_type = attention_type

        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )

        self.roberta_encoder = AutoModel.from_pretrained(
            model_path, config=self.config, trust_remote_code=True
        )
        if attention_type == "dual_encoding":
            self.roberta_encoder.gradient_checkpointing_enable()

        # For two-stage retrieval: add frozen encoder for Stage 1
        if attention_type == "token_level":
            print("Creating frozen encoder for Stage 1 retrieval")
            self.frozen_encoder = AutoModel.from_pretrained(
                model_path, config=self.config, trust_remote_code=True
            )
            # Freeze all parameters
            for param in self.frozen_encoder.parameters():
                param.requires_grad = False

        if cross_attention:
            encoder_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            if attention_type == "token_level":
                print("Using TokenLevelCrossAttention")
                # self.token_level_attention = TokenLevelDescriptionCrossAttention(
                #     input_size=self.config.hidden_size, 
                #     num_classes=num_classes, 
                #     scale=scale,
                #     encoder_model=self.roberta_encoder,
                #     encoder_tokenizer=encoder_tokenizer,
                #     target_tokenizer=target_tokenizer,
                #     icd_version=kwargs.get('icd_version', 10),
                #     desc_batch_size=kwargs.get('desc_batch_size', 64),
                #     max_desc_len=kwargs.get('max_desc_len', 64),
                # )
    
                self.label_wise_attention = LabelCrossAttention(
                    input_size=self.config.hidden_size, num_classes=num_classes, scale=scale
                )
                
                # Dynamic token-level attention for two-stage retrieval
                self._dynamic_token_attention = DynamicTokenLevelCrossAttention(
                    input_size=self.config.hidden_size,
                    scale=scale,
                    encoder_tokenizer=encoder_tokenizer,
                    target_tokenizer=target_tokenizer,
                    icd_version=kwargs.get('icd_version', 10),
                    max_desc_len=kwargs.get('max_desc_len', 64),
                    pooling_temperature=kwargs.get('pooling_temperature', 1.0),
                    random_init=random_init
                )
            elif attention_type == "dual_encoding":
                self.label_wise_attention = LabelCrossAttentionDE(
                    input_size=self.config.hidden_size, 
                    num_classes=num_classes, 
                    scale=scale,
                    encoder_model=self.roberta_encoder,
                    encoder_tokenizer=encoder_tokenizer,
                    target_tokenizer=target_tokenizer,
                    icd_version=kwargs.get('icd_version', 10),
                    desc_batch_size=kwargs.get('desc_batch_size', 64),
                    random_init=random_init,
                    # model_path=model_path,
                    # init_with_descriptions=kwargs.get('init_with_descriptions', True),
                    # freeze_label_embeddings=kwargs.get('freeze_label_embeddings', False)
                )
            else:
                self.label_wise_attention = LabelCrossAttention(
                    input_size=self.config.hidden_size, 
                    num_classes=num_classes, 
                    scale=scale, 
                    target_tokenizer=target_tokenizer,  
                    icd_version=kwargs.get('icd_version', 10),
                    desc_batch_size=kwargs.get('desc_batch_size', 64),
                    random_init=random_init,
                    model_path=model_path,
                    init_with_descriptions=init_with_descriptions,
                    freeze_label_embeddings=freeze_label_embeddings
                )

        else:
            self.label_wise_attention = LabelAttention(
                input_size=self.config.hidden_size,
                projection_size=self.config.hidden_size,
                num_classes=num_classes,
            )
        self.mask_input = mask_input
        if self.mask_input:
            self.input_masker = InputMasker(
                input_size=self.config.hidden_size, scale=scale
            )

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(input_ids=input_ids, attention_masks=attention_mask)
        )

    def split_input_into_chunks(
        self, input_sequence: torch.Tensor, pad_index: int
    ) -> torch.Tensor:
        """Split input into chunks of chunk_size.

        Args:
            input_sequence (torch.Tensor): input sequence to split (batch_size, seq_len)
            pad_index (int): padding index

        Returns:
            torch.Tensor: reshaped input (batch_size, num_chunks, chunk_size)
        """
        batch_size = input_sequence.size(0)
        # pad input to be divisible by chunk_size
        input_sequence = nn.functional.pad(
            input_sequence,
            (0, self.chunk_size - input_sequence.size(1) % self.chunk_size),
            value=pad_index,
        )
        return input_sequence.view(batch_size, -1, self.chunk_size)

    def roberta_encode_embedding_input(self, embedding, attention_masks):
        input_shape = embedding.size()[:-1]
        extended_attention_mask = self.roberta_encoder.get_extended_attention_mask(
            attention_masks, input_shape
        )
        head_mask = self.roberta_encoder.get_head_mask(
            None, self.roberta_encoder.config.num_hidden_layers
        )
        encoder_outputs = self.roberta_encoder.encoder(
            embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        return sequence_output

    def get_chunked_attention_masks(
        self, attention_masks: torch.Tensor
    ) -> torch.Tensor:
        return self.split_input_into_chunks(attention_masks, 0)

    def get_input_embeddings(self):
        return self.roberta_encoder.embeddings

    def get_chunked_embedding(self, input_ids):
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        embedding = self.roberta_encoder.embeddings(input_ids.view(-1, chunk_size))
        return embedding

    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings. Huggingface Roberta model can't return more than 512 token embeddings at once.

        Args:
            input_ids (torch.Tensor): input ids

        Returns:
            torch.Tensor: token embeddings
        """
        sequence_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        if sequence_length <= 512:
            return self.roberta_encoder.embeddings(input_ids)
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        chunked_embeddings = self.roberta_encoder.embeddings(
            input_ids.view(-1, chunk_size)
        )
        embeddings = chunked_embeddings.view(
            batch_size, -1, chunked_embeddings.size(-1)
        )
        return embeddings[:, :sequence_length]

    def _frozen_encoder(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Use frozen encoder for Stage 1 retrieval."""
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks = self.get_chunked_attention_masks(attention_masks)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.frozen_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks.view(-1, chunk_size)
            if attention_masks is not None
            else None,
            return_dict=False,
        )
        return outputs[0].view(batch_size, num_chunks * chunk_size, -1)

    def _encode_descriptions(self) -> torch.Tensor:
        """Encode all descriptions at once (no batching)."""
        with torch.set_grad_enabled(self.training):
            desc_outputs = self.label_wise_attention.encoder_model(
                input_ids=self.label_wise_attention.description_input_ids,
                attention_mask=self.label_wise_attention.description_attention_mask
            )

            # Check if this is TokenLevelCrossAttention
            if isinstance(self.label_wise_attention, TokenLevelDescriptionCrossAttention):
                # Return token-level embeddings for token-level attention
                return desc_outputs.last_hidden_state
            else:
                # Mean pooling with attention mask for LabelCrossAttentionDE
                attention_mask = self.label_wise_attention.description_attention_mask.unsqueeze(-1)
                masked_embeddings = desc_outputs.last_hidden_state * attention_mask
                all_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1)
                return all_embeddings

    def _encode_subset_descriptions(self, top_k_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode only the descriptions for the given top-k indices."""
        batch_size, k = top_k_indices.shape
        
        # Handle batched top_k_indices by processing each batch item
        all_encoded_descriptions = []
        all_attention_masks = []
        
        for batch_idx in range(batch_size):
            batch_indices = top_k_indices[batch_idx]  # [k]
            
            subset_input_ids = self._dynamic_token_attention.description_input_ids[batch_indices]  # [k, seq_len]
            subset_attention_mask = self._dynamic_token_attention.description_attention_mask[batch_indices]  # [k, seq_len]
            
            # Encode subset of descriptions using the trainable encoder
            with torch.set_grad_enabled(self.training):
                desc_outputs = self.roberta_encoder(
                    input_ids=subset_input_ids,
                    attention_mask=subset_attention_mask
                )
                
                all_encoded_descriptions.append(desc_outputs.last_hidden_state)  # [k, desc_seq, dim]
                all_attention_masks.append(subset_attention_mask)  # [k, desc_seq]
        
        # Stack to get [batch_size, k, desc_seq, dim] and [batch_size, k, desc_seq]
        encoded_descriptions = torch.stack(all_encoded_descriptions, dim=0)
        attention_masks = torch.stack(all_attention_masks, dim=0)
        
        return encoded_descriptions, attention_masks


    def expand_topk_to_full_logits(
        self, 
        top_k_logits: torch.Tensor, 
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Expand top-k logits to full class space for compatibility with metrics."""
        batch_size = top_k_logits.size(0)
        # Use large negative value instead of -inf to avoid numerical issues  
        # Use float32 to avoid BFloat16 conversion issues
        full_logits = torch.full(
            (batch_size, self.num_classes), 
            -1e9, 
            device=top_k_logits.device, 
            dtype=torch.float32
        )
        full_logits.scatter_(dim=1, index=top_k_indices, src=top_k_logits.float())
        return full_logits

    def encoder(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks = self.get_chunked_attention_masks(attention_masks)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks.view(-1, chunk_size)
            if attention_masks is not None
            else None,
            return_dict=False,
        )
        return outputs[0].view(batch_size, num_chunks * chunk_size, -1)

    def forward_with_input_masking(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_mask: bool = False,
        baseline_token_id: int = 500001,
    ):
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=baseline_token_id,
            eos_token_id=baseline_token_id,
        )
        baseline = baseline.requires_grad_(False)
        baseline_embeddings = self.get_chunked_embedding(baseline).detach()
        chunked_input_embeddings = self.get_chunked_embedding(input_ids)
        with torch.no_grad():
            token_representations = (
                self.get_token_representations_from_chunked_embeddings(
                    chunked_input_embeddings.detach(), attention_masks
                )
            )
        input_mask = self.input_masker(
            token_representations, attention_masks=attention_masks
        )

        input_mask_sigmoid = torch.sigmoid(input_mask)
        input_mask_sigmoid = input_mask_sigmoid.view(-1, self.chunk_size, 1)
        masked_chunked_input_embeddings = (
            chunked_input_embeddings * input_mask_sigmoid
            + baseline_embeddings * (1 - input_mask_sigmoid)
        )
        masked_token_representations = (
            self.get_token_representations_from_chunked_embeddings(
                masked_chunked_input_embeddings, attention_masks
            )
        )
        # Encode descriptions if using cross attention
        encoded_descriptions = None
        if hasattr(self.label_wise_attention, 'encoder_model'):
            encoded_descriptions = self._encode_descriptions()
        
        if output_mask:
            return self.label_wise_attention(
                masked_token_representations,
                attention_masks=attention_masks,
                output_attention=output_attentions,
                encoded_descriptions=encoded_descriptions,
            ), input_mask

        return self.label_wise_attention(
            masked_token_representations,
            attention_masks=attention_masks,
            output_attention=output_attentions,
            encoded_descriptions=encoded_descriptions,
        )

    def get_token_representations_from_chunked_embeddings(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Get token representations from chunked embeddings.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Token representations of shape [batch_size, sequence_length, hidden_size]
        """
        (
            num_chunks_times_batch_size,
            chunk_size,
            embedding_size,
        ) = chunked_embedding.size()
        batch_size = attention_masks.size(0)
        num_chunks = num_chunks_times_batch_size // batch_size
        chunked_attention_masks = self.get_chunked_attention_masks(attention_masks)
        hidden_outputs = self.roberta_encode_embedding_input(
            embedding=chunked_embedding.view(-1, chunk_size, embedding_size),
            attention_masks=chunked_attention_masks.view(-1, chunk_size),
        )
        return hidden_outputs.view(batch_size, num_chunks * chunk_size, -1)

    def forward_embedding_input(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
        output_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the model with chunked embedding input.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, num_chunks, chunk_size]

        Returns:
            torch.Tensor:
        """
        token_representations = self.get_token_representations_from_chunked_embeddings(
            chunked_embedding, attention_masks
        )
        # Encode descriptions if using cross attention
        encoded_descriptions = None
        if hasattr(self.label_wise_attention, 'encoder_model'):
            encoded_descriptions = self._encode_descriptions()
        
        return self.label_wise_attention(
            token_representations,
            attention_masks=attention_masks,
            output_attention=output_attention,
            encoded_descriptions=encoded_descriptions,
        )

    @torch.no_grad()
    def de_chunk_attention(
        self,
        attentions_chunked: torch.Tensor,
        batch_size: int,
        num_layers: int,
        num_chunks: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """De-chunk attention.

        Args:
            attentions_chunked (torch.Tensor): Attention matrix of shape [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
            batch_size (int): Batch size
            num_layers (int): Number of layers
            num_chunks (int): Number of chunks
            chunk_size (int): Chunk size

        Returns:
            torch.Tensor: Attention matrix of shape [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]
        """
        attentions = torch.zeros(
            batch_size,
            num_layers,
            num_chunks * chunk_size,
            num_chunks * chunk_size,
            device=attentions_chunked.device,
            dtype=torch.float16,
        )
        for chunk_idx in range(num_chunks):
            attentions[
                :,
                :,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
            ] = attentions_chunked[:, chunk_idx]

        return attentions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
        top_k: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.mask_input:
            return self.forward_with_input_masking(
                input_ids, attention_masks, output_attentions, False
            )
        
        if top_k is not None:
            # Stage 1: Use frozen encoder for retrieval
            with torch.no_grad():
                frozen_hidden_output = self._frozen_encoder(input_ids, attention_masks)
                logits = self.label_wise_attention(
                    frozen_hidden_output,
                    attention_masks=attention_masks,
                    output_attention=False,
                )
                
                # Get top-k indices from logits
                top_k_scores, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                
                # During training, ensure all true targets are included
                if self.training:
                    batch_size = targets.size(0)
                    for batch_idx in range(batch_size):
                        # Get true target indices for this sample
                        true_targets = targets[batch_idx].nonzero(as_tuple=False).squeeze(-1)
                        if len(true_targets) > 0:
                            # Start with true targets, then add top-k predictions (filtering out duplicates)
                            mask = ~torch.isin(top_k_indices[batch_idx], true_targets)
                            remaining_predictions = top_k_indices[batch_idx][mask]
                            needed = top_k - len(true_targets)
                            final_indices = torch.cat([true_targets, remaining_predictions[:needed]])
                            top_k_indices[batch_idx] = final_indices
            
            # Stage 2: Use trainable encoder for token-level attention
            trainable_hidden_output = self.encoder(input_ids, attention_masks)
            top_k_encoded_descriptions, top_k_desc_mask = self._encode_subset_descriptions(top_k_indices)
            
            top_k_logits = self._dynamic_token_attention(
                trainable_hidden_output,
                top_k_encoded_descriptions,
                top_k_desc_mask,
                attention_masks,
            )
            
            # Expand top-k logits to full class space for compatibility
            full_logits = self.expand_topk_to_full_logits(top_k_logits, top_k_indices)
            return full_logits
        
        # Normal forward pass
        hidden_output = self.encoder(input_ids, attention_masks)

        if self.attention_type == "dual_encoding":
            encoded_descriptions = self._encode_descriptions()
            return self.label_wise_attention(
                hidden_output,
                attention_masks=attention_masks,
                output_attention=output_attentions,
                attn_grad_hook_fn=attn_grad_hook_fn,
                encoded_descriptions=encoded_descriptions,
            )
        else:
            return self.label_wise_attention(
                hidden_output,
                attention_masks=attention_masks,
                output_attention=output_attentions,
                attn_grad_hook_fn=attn_grad_hook_fn,
            )

    @torch.no_grad()
    def get_encoder_attention_and_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks_chunks = self.split_input_into_chunks(attention_masks, 0)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks_chunks.view(-1, chunk_size)
            if attention_masks_chunks is not None
            else None,
            return_dict=False,
            output_attentions=True,
            output_hidden_states=True,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        _, label_wise_attention = self.label_wise_attention(
            hidden_output, attention_masks, True
        )
        return outputs[2], outputs[3], label_wise_attention

    @torch.no_grad()
    def attention_rollout(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            _,
            attentions,
            label_wise_attention,
        ) = self.get_encoder_attention_and_hidden_states(input_ids, attention_masks)

        label_wise_attention = torch.softmax(label_wise_attention, dim=1)
        attentions = (
            torch.stack(attentions).to(torch.float16).transpose(1, 0)
        )  # [batch_size*num_chunks, num_layers, num_heads, chunk_size, chunk_size]

        batch_size = input_ids.size(0)
        num_chunks = attentions.size(0) // batch_size
        num_layers = attentions.size(1)
        num_heads = attentions.size(2)
        chunk_size = attentions.size(3)

        attentions = attentions.view(
            batch_size, num_chunks, num_layers, num_heads, chunk_size, chunk_size
        )
        attentions = torch.mean(
            attentions, dim=3
        )  # [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
        attentions = self.de_chunk_attention(
            attentions, batch_size, num_layers, num_chunks, chunk_size
        )  # [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]

        attentions = (
            attentions
            + torch.eye(chunk_size * num_chunks, device=attentions.device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 2
        )  # add skip connection

        attention_rollout = attentions[:, 0]
        for hidden_layer_idx in range(1, num_layers):
            attention_rollout = attentions[:, hidden_layer_idx] @ attention_rollout

        return label_wise_attention @ attention_rollout
