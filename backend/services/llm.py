"""
LLM Service

Handles communication with the local LLM API endpoint.
"""

import json
import requests
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for communicating with a local LLM API.
    
    This class handles requests to a locally hosted LLM API that follows
    the OpenAI API format.
    """
    
    def __init__(
        self,
        api_endpoint: str = "http://127.0.0.1:1234/v1/chat/completions",
        model: str = "phi-4-q8_0",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 60
    ):
        """
        Initialize the LLM client.
        
        Args:
            api_endpoint: URL of the local LLM API
            model: Model name to use (or 'default' for API default)
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.api_endpoint = api_endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # State tracking
        self.is_processing = False
        self.conversation_history = []
        
        logger.info(f"Initialized LLM Client with endpoint={api_endpoint}")
        
    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            role: Message role ('system', 'user', or 'assistant')
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        
        # Allow deeper history for models with large context windows
        if len(self.conversation_history) > 50:
            # Always keep the system message if it exists
            if self.conversation_history[0]["role"] == "system":
                self.conversation_history = (
                    [self.conversation_history[0]] + 
                    self.conversation_history[-49:]
                )
            else:
                self.conversation_history = self.conversation_history[-50:]
    
    def get_response(self, user_input: str, system_prompt: Optional[str] = None, 
                    add_to_history: bool = True, temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Get a response from the LLM for the given user input.
        
        Args:
            user_input: User's text input
            system_prompt: Optional system prompt to set context
            add_to_history: Whether to add this exchange to conversation history
            temperature: Optional temperature override (0.0 to 1.0)
            
        Returns:
            Dictionary containing the LLM response and metadata
        """
        self.is_processing = True
        start_time = logging.Formatter.converter()
        
        try:
            # Prepare messages
            messages = []
            
            # Add system prompt if provided and not already in history
            if system_prompt:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add user input to history if it's not empty and add_to_history is True
            if user_input.strip() and add_to_history:
                self.add_to_history("user", user_input)
            
            # Add conversation history (which now includes the user input if add_to_history=True)
            messages.extend(self.conversation_history)
            
            # Only add user input directly if not adding to history
            # This ensures special cases (greetings/followups) work while preventing duplication for normal speech
            if user_input.strip() and not add_to_history:
                messages.append({
                    "role": "user",
                    "content": user_input
                })
            
            # Prepare request payload with custom temperature if provided
            payload = {
                "model": self.model if self.model != "default" else None,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Remove None values
            payload = {k: v for k, v in payload.items() if v is not None}
            
            # Log the full payload (truncated for readability)
            payload_str = json.dumps(payload)
            logger.info(f"Sending request to LLM API with {len(messages)} messages")
            
            # Add more detailed logging to help debug message duplication
            message_roles = [msg["role"] for msg in messages]
            user_message_count = message_roles.count("user")
            logger.info(f"Message roles: {message_roles}, user messages: {user_message_count}")
            
            if len(payload_str) > 500:
                logger.debug(f"Payload (truncated): {payload_str[:500]}...")
            else:
                logger.debug(f"Payload: {payload_str}")
            
            # Send request to LLM API
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            # Check if request was successful
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract assistant response
            assistant_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Add assistant response to history (only if we added the user input)
            if assistant_message and add_to_history:
                self.add_to_history("assistant", assistant_message)
            
            # Calculate processing time
            end_time = logging.Formatter.converter()
            processing_time = end_time[0] - start_time[0]
            
            logger.info(f"Received response from LLM API after {processing_time:.2f}s")
            
            return {
                "text": assistant_message,
                "processing_time": processing_time,
                "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
                "model": result.get("model", "unknown")
            }
            
        except requests.RequestException as e:
            logger.error(f"LLM API request error: {e}")
            error_response = f"I'm sorry, I encountered a problem connecting to my language model. {str(e)}"
            
            # Add the error to history if requested and clear history on 400 errors
            # to prevent the same error from happening repeatedly
            if add_to_history:
                self.add_to_history("assistant", error_response)
                
                # If we get a 400 Bad Request, the context might be corrupt
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 400:
                    logger.warning("Received 400 error, clearing conversation history to recover")
                    # Keep only system prompt if it exists
                    self.clear_history(keep_system_prompt=True)
            
            return {
                "text": error_response,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"LLM processing error: {e}")
            error_response = "I'm sorry, I encountered an unexpected error. Please try again."
            self.add_to_history("assistant", error_response)
            return {
                "text": error_response,
                "error": str(e)
            }
        finally:
            self.is_processing = False
    
    def clear_history(self, keep_system_prompt: bool = True) -> None:
        """
        Clear conversation history.
        
        Args:
            keep_system_prompt: Whether to keep the system prompt if it exists
        """
        if keep_system_prompt and self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history = [self.conversation_history[0]]
        else:
            self.conversation_history = []
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict containing the current configuration
        """
        return {
            "api_endpoint": self.api_endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "is_processing": self.is_processing,
            "history_length": len(self.conversation_history)
        }
