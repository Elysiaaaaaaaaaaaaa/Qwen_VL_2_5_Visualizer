"""
Attention Extraction Module for Qwen2.5-VL

This module provides functionality to hook into the Qwen2.5-VL model
and extract attention weights during text generation.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import config


class AttentionHook:
    """
    Hook class to capture attention weights from a specific layer.
    """

    def __init__(
        self,
        layer_idx: int,
        store_on_cpu: bool = config.STORE_ON_CPU,
        use_float16: bool = config.USE_FLOAT16
    ):
        """
        Initialize attention hook.

        Args:
            layer_idx: Index of the layer this hook is attached to
            store_on_cpu: Whether to move attention to CPU for storage
            use_float16: Whether to convert to float16 for storage
        """
        self.layer_idx = layer_idx
        self.store_on_cpu = store_on_cpu
        self.use_float16 = use_float16
        self.attention_weights = {}
        self.hidden_states = {}
        self.generation_step = 0

    def __call__(self, module, input, output):
        """
        Hook function called during forward pass.

        Args:
            module: The attention module
            input: Input to the module
            output: Output from the module (hidden_states, attention_weights)
        """
        # Output format: (attn_output, attn_weights) or just attn_output
        # For Qwen2.5-VL, we need to check if attention weights are returned
        if isinstance(output, tuple) and len(output) >= 2:
            attn_output = output[0]  # This is the hidden state
            attn_weights = output[1]

            # Store hidden state
            if attn_output is not None:
                if self.use_float16:
                    attn_output = attn_output.half()
                if self.store_on_cpu:
                    attn_output = attn_output.cpu()
                # Use key_len (total sequence length) to determine generation step
                step_key = attn_output.shape[1]  # Shape: (batch_size, seq_len, hidden_dim)
                self.hidden_states[step_key] = attn_output[0].clone()  # Store for batch 0

            if attn_weights is not None:
                # Store attention weights
                # Shape: (batch_size, num_heads, seq_len, seq_len) or (batch_size, num_heads, 1, seq_len)
                if self.use_float16:
                    attn_weights = attn_weights.half()

                if self.store_on_cpu:
                    attn_weights = attn_weights.cpu()

                # Store per head
                batch_size, num_heads, query_len, key_len = attn_weights.shape
                
                # Use key_len (total sequence length) to determine generation step
                # In autoregressive generation, each new token sees all previous tokens
                # So generation_step = key_len - input_length
                # For now, use key_len as a unique identifier
                step_key = key_len
                
                for head_idx in range(num_heads):
                    key = (step_key, head_idx)
                    self.attention_weights[key] = attn_weights[0, head_idx].clone()

    def get_attention_weights(self, generation_step: Optional[int] = None) -> Dict[str, Any]:
        """
        Get stored attention weights for all heads at a specific generation step.

        Args:
            generation_step: Which generation step to retrieve (None = last step)

        Returns:
            Dictionary containing 'attention' (head_idx to attention tensor) and 'hidden_state'
        """
        if generation_step is None:
            # Get the latest step by finding the maximum step_key
            if self.attention_weights:
                generation_step = max(step for step, _ in self.attention_weights.keys())
            else:
                generation_step = 0
            
        current_step_attn = {}
        for (step, head_idx), attn in self.attention_weights.items():
            if step == generation_step:
                current_step_attn[head_idx] = attn

        # Get hidden state for the same generation step
        hidden_state = self.hidden_states.get(generation_step, None)

        return {
            'attention': current_step_attn,
            'hidden_state': hidden_state
        }
    
    def get_all_attention_weights(self) -> Tuple[Dict[int, Dict[int, torch.Tensor]], Dict[int, torch.Tensor]]:
        """
        Get attention weights for all generation steps, organized by step then head.
        
        Returns:
            Tuple of (attention_dict, hidden_state_dict):
            - attention_dict: {step: {head_idx: attention_tensor}}
            - hidden_state_dict: {step: hidden_state_tensor}
        """
        attention_dict = {}
        hidden_state_dict = {}
        
        for (step, head_idx), attn in self.attention_weights.items():
            if step not in attention_dict:
                attention_dict[step] = {}
            attention_dict[step][head_idx] = attn
        
        # Collect hidden states
        for step, hidden in self.hidden_states.items():
            hidden_state_dict[step] = hidden
            
        return attention_dict, hidden_state_dict

    def increment_step(self):
        """Increment the generation step counter."""
        self.generation_step += 1

    def reset(self):
        """Clear stored attention weights and hidden states."""
        self.attention_weights.clear()
        self.hidden_states.clear()
        self.generation_step = 0


class AttentionExtractor:
    """
    Main class for extracting attention from Qwen2.5-VL model.
    """

    def __init__(
        self,
        model,
        extract_all_layers: bool = config.EXTRACT_ALL_LAYERS,
        specific_layers: Optional[List[int]] = None,
        store_on_cpu: bool = config.STORE_ON_CPU,
        use_float16: bool = config.USE_FLOAT16
    ):
        """
        Initialize attention extractor.

        Args:
            model: Qwen2.5-VL model instance
            extract_all_layers: Whether to extract from all layers
            specific_layers: List of specific layer indices to extract from
            store_on_cpu: Whether to store attention on CPU
            use_float16: Whether to use float16 for storage
        """
        self.model = model
        self.extract_all_layers = extract_all_layers
        self.specific_layers = specific_layers if specific_layers else config.SPECIFIC_LAYERS
        self.store_on_cpu = store_on_cpu
        self.use_float16 = use_float16

        self.hooks = {}
        self.hook_handles = []
        self.is_active = False

        # Determine which layers to hook
        self.target_layers = self._determine_target_layers()

    def _determine_target_layers(self) -> List[int]:
        """
        Determine which layers to extract attention from.

        Returns:
            List of layer indices
        """
        # Get total number of layers
        num_layers = len(self.model.model.language_model.layers)

        if self.extract_all_layers:
            return list(range(num_layers))
        else:
            # Use specific layers, ensuring they're within range
            return [idx for idx in self.specific_layers if idx < num_layers]

    def _register_hooks(self):
        """Register forward hooks on attention modules."""
        if self.is_active:
            return

        # Navigate to decoder layers
        # Path: model.model.language_model.layers[i].self_attn
        for layer_idx in self.target_layers:
            layer = self.model.model.language_model.layers[layer_idx]
            attention_module = layer.self_attn

            # Create hook for this layer
            hook = AttentionHook(
                layer_idx=layer_idx,
                store_on_cpu=self.store_on_cpu,
                use_float16=self.use_float16
            )
            self.hooks[layer_idx] = hook

            # Register the hook
            handle = attention_module.register_forward_hook(hook)
            self.hook_handles.append(handle)

        self.is_active = True

        if config.VERBOSE:
            print(f"Registered attention hooks on {len(self.target_layers)} layers: {self.target_layers}")

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()

        self.hook_handles.clear()
        self.is_active = False

        if config.VERBOSE:
            print("Removed all attention hooks")

    def enable_attention_output(self):
        """
        Enable attention output in the model.
        This is required to capture attention weights.
        """
        # Set output_attentions=True in model config
        self.model.config.output_attentions = True
        self.model.model.language_model.config.output_attentions = True

        if config.VERBOSE:
            print("Enabled attention output in model config")

    def disable_attention_output(self):
        """Disable attention output to save memory."""
        self.model.config.output_attentions = False
        self.model.model.language_model.config.output_attentions = False

    def start_extraction(self):
        """Start extracting attention weights."""
        self.enable_attention_output()
        self._register_hooks()

    def stop_extraction(self):
        """Stop extracting attention weights."""
        self._remove_hooks()
        self.disable_attention_output()

    def get_attention_weights(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all extracted attention weights from the last generation step.

        Returns:
            Dictionary: {layer_idx: {'attention': {head_idx: attention_tensor}, 'hidden_state': hidden_state}}
        """
        attention_dict = {}

        for layer_idx, hook in self.hooks.items():
            attention_dict[layer_idx] = hook.get_attention_weights()

        return attention_dict
    
    def get_attention_for_generation_step(self, step: int) -> Dict[int, Dict[str, Any]]:
        """
        Get attention weights for a specific generation step.
        
        Args:
            step: Generation step index
            
        Returns:
            Dictionary: {layer_idx: {'attention': {head_idx: attention_tensor}, 'hidden_state': hidden_state}}
        """
        attention_dict = {}
        
        for layer_idx, hook in self.hooks.items():
            attention_dict[layer_idx] = hook.get_attention_weights(generation_step=step)
        
        return attention_dict
    
    def get_all_generation_steps(self) -> Tuple[Dict[int, Dict[int, Dict[int, torch.Tensor]]], Dict[int, Dict[int, torch.Tensor]]]:
        """
        Get attention weights for all generation steps.
        
        Returns:
            Tuple of (attention_dict, hidden_state_dict):
            - attention_dict: {step: {layer_idx: {head_idx: attention_tensor}}}
            - hidden_state_dict: {step: {layer_idx: hidden_state_tensor}}
        """
        # First get all attention by step from hooks
        all_steps_attention = {}
        all_steps_hidden = {}
        
        for layer_idx, hook in self.hooks.items():
            layer_attention, layer_hidden = hook.get_all_attention_weights()
            
            # Organize attention by step
            for step, head_dict in layer_attention.items():
                if step not in all_steps_attention:
                    all_steps_attention[step] = {}
                all_steps_attention[step][layer_idx] = head_dict
            
            # Organize hidden states by step
            for step, hidden in layer_hidden.items():
                if step not in all_steps_hidden:
                    all_steps_hidden[step] = {}
                all_steps_hidden[step][layer_idx] = hidden
        
        return all_steps_attention, all_steps_hidden

    def get_attention_for_token(
        self,
        token_position: int,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> Optional[torch.Tensor]:
        """
        Get attention weights for a specific token position.

        Args:
            token_position: Position of the token in the sequence
            layer_idx: Specific layer (None = all layers)
            head_idx: Specific head (None = all heads)

        Returns:
            Attention tensor or None if not found
        """
        all_attention = self.get_attention_weights()

        if layer_idx is not None:
            if layer_idx not in all_attention:
                return None

            layer_data = all_attention[layer_idx]
            layer_attention = layer_data['attention']

            if head_idx is not None:
                if head_idx not in layer_attention:
                    return None
                # Return attention from specific token to all previous tokens
                return layer_attention[head_idx][token_position, :]
            else:
                # Return attention from all heads
                return torch.stack([
                    attn[token_position, :]
                    for attn in layer_attention.values()
                ])
        else:
            # Return attention from all layers and heads
            all_token_attention = []
            for layer_data in all_attention.values():
                layer_attention = layer_data['attention']
                for head_attn in layer_attention.values():
                    all_token_attention.append(head_attn[token_position, :])

            if all_token_attention:
                return torch.stack(all_token_attention)
            return None

    def reset(self):
        """Reset all stored attention weights and hidden states."""
        for hook in self.hooks.values():
            hook.reset()

        if config.VERBOSE:
            print("Reset all attention weights and hidden states")

    def __enter__(self):
        """Context manager entry."""
        self.start_extraction()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_extraction()


def extract_attention_during_generation(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    **kwargs
) -> Tuple[torch.Tensor, Dict[int, Dict[int, torch.Tensor]]]:
    """
    Generate text while extracting attention weights.

    Args:
        model: Qwen2.5-VL model
        input_ids: Input token IDs
        attention_mask: Attention mask
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        **kwargs: Additional generation parameters

    Returns:
        Tuple of (generated_ids, attention_weights_dict)
    """
    extractor = AttentionExtractor(model)

    with extractor:
        # Generate with attention extraction
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            output_attentions=True,
            return_dict_in_generate=True,
            **kwargs
        )

        generated_ids = output.sequences

        # Get extracted attention
        attention_weights = extractor.get_attention_weights()

    return generated_ids, attention_weights


def create_attention_extractor(
    model,
    extract_all: bool = True,
    specific_layers: Optional[List[int]] = None
) -> AttentionExtractor:
    """
    Convenience function to create an AttentionExtractor.

    Args:
        model: Qwen2.5-VL model
        extract_all: Extract from all layers
        specific_layers: Specific layers to extract from

    Returns:
        AttentionExtractor instance
    """
    return AttentionExtractor(
        model=model,
        extract_all_layers=extract_all,
        specific_layers=specific_layers
    )
