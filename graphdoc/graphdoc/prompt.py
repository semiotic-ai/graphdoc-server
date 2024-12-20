# system packages
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any, Dict

# internal packages

# external packages

@dataclass
class RequestObject: 
    prompt: str
    response: str
    model: str
    prompt_tokens: int
    response_tokens: int
    request_time: int
    request_id: str 
    request_object: Any

@dataclass
class PromptRevision:
    content: str
    revision_number: int
    author: str
    timestamp: datetime = field(default_factory=datetime.now)
    comments: Optional[str] = None
    previous_revision: Optional["PromptRevision"] = None
    base_prompt: Optional[str] = None
    request_object: Optional[RequestObject] = None  

@dataclass
class Prompt:
    title: str
    base_content: str
    revisions: List[PromptRevision] = field(default_factory=list)
    current_revision: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_revision(
        self,
        content: str,
        author: str,
        comments: Optional[str] = None,
        base_prompt: Optional[str] = None,
        request_object: Optional[RequestObject] = None
    ) -> PromptRevision:
        """
        Add a new revision to the prompt.
        Returns the created PromptRevision object.
        """
        self.current_revision += 1
        previous = self.revisions[-1] if self.revisions else None
        
        revision = PromptRevision(
            content=content,
            revision_number=self.current_revision,
            author=author,
            comments=comments,
            previous_revision=previous,
            base_prompt=base_prompt or self.base_content,
            request_object=request_object
        )
        
        self.revisions.append(revision)
        return revision
    
    @property
    def current_content(self) -> Optional[str]:
        """Get the content of the latest revision"""
        if self.revisions:
            return self.revisions[-1].content
        return self.base_content
    
    def get_revision(self, revision_number: int) -> Optional[PromptRevision]:
        """Get a specific revision by number"""
        for revision in self.revisions:
            if revision.revision_number == revision_number:
                return revision
        return None
    
    def get_revision_history(self) -> List[dict]:
        """Get complete revision history with all metadata"""
        return [
            {
                'revision': rev.revision_number,
                'content': rev.content,
                'timestamp': rev.timestamp,
                'author': rev.author,
                'comments': rev.comments,
                'base_prompt': rev.base_prompt,
                'request_details': {
                    'model': rev.request_object.model,
                    'prompt_tokens': rev.request_object.prompt_tokens,
                    'response_tokens': rev.request_object.response_tokens,
                    'request_time': rev.request_object.request_time,
                    'request_id': rev.request_object.request_id
                } if rev.request_object else None
            }
            for rev in self.revisions
        ]
    
    def save_request(
            self,
            prompt: str,
            response_text: str,
            model: str,
            prompt_tokens: int,
            response_tokens: int,
            request_time: int,
            request_id: str,
            author: str,
            raw_response: Any = None,
            comments: Optional[str] = None
        ) -> PromptRevision:
            """
            Save a request and its response as a new revision
            
            Args:
                prompt: The input prompt text
                response_text: The generated response text
                model: Name/identifier of the model used
                prompt_tokens: Number of tokens in prompt
                response_tokens: Number of tokens in response
                request_time: Timestamp of request
                request_id: Unique identifier for request
                author: Who/what made the request
                raw_response: Optional raw response object
                comments: Optional comments about this revision
            """
            request_obj = RequestObject(
                prompt=prompt,
                response=response_text,
                model=model,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                request_time=request_time,
                request_id=request_id,
                request_object=raw_response
            )
            
            return self.add_revision(
                content=response_text,
                author=author,
                comments=comments,
                base_prompt=prompt,
                request_object=request_obj
            )
    
    def get_latest_request(self) -> Optional[RequestObject]:
        """Get the most recent request object"""
        if self.revisions:
            return self.revisions[-1].request_object
        return None