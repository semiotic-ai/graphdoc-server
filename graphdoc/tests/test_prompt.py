# system packages
from datetime import datetime
from typing import Dict, Any

# internal packages
from graphdoc import Prompt, PromptRevision, RequestObject

# external packages
import pytest


class TestPrompt:
    def test_prompt_initialization(self, sample_prompt):
        assert sample_prompt.title == "Test Prompt"
        assert sample_prompt.base_content == "This is a base prompt content"
        assert sample_prompt.current_revision == 0
        assert len(sample_prompt.revisions) == 0
        assert sample_prompt.metadata == {"type": "test"}

    def test_add_revision(self, sample_prompt):
        revision = sample_prompt.add_revision(
            content="New content", author="test_author", comments="Test comment"
        )

        assert len(sample_prompt.revisions) == 1
        assert sample_prompt.current_revision == 1
        assert revision.content == "New content"
        assert revision.author == "test_author"
        assert revision.comments == "Test comment"
        assert revision.revision_number == 1
        assert revision.previous_revision is None
        assert revision.base_prompt == sample_prompt.base_content

    def test_multiple_revisions(self, sample_prompt):
        # Add first revision
        rev1 = sample_prompt.add_revision(content="First revision", author="author1")

        # Add second revision
        rev2 = sample_prompt.add_revision(content="Second revision", author="author2")

        assert len(sample_prompt.revisions) == 2
        assert sample_prompt.current_revision == 2
        assert rev2.previous_revision == rev1

    def test_current_content(self, sample_prompt):
        # Test with no revisions
        assert sample_prompt.current_content == sample_prompt.base_content

        # Test with revision
        sample_prompt.add_revision(content="New content", author="test_author")
        assert sample_prompt.current_content == "New content"

    def test_get_revision(self, sample_prompt):
        revision = sample_prompt.add_revision(
            content="Test content", author="test_author"
        )

        assert sample_prompt.get_revision(1) == revision
        assert sample_prompt.get_revision(999) is None

    def test_get_revision_history(self, sample_prompt, sample_request_object):
        sample_prompt.add_revision(
            content="Test content",
            author="test_author",
            comments="Test comment",
            request_object=sample_request_object,
        )

        history = sample_prompt.get_revision_history()
        assert len(history) == 1

        revision_data = history[0]
        assert revision_data["revision"] == 1
        assert revision_data["content"] == "Test content"
        assert revision_data["author"] == "test_author"
        assert revision_data["comments"] == "Test comment"
        assert revision_data["request_details"]["model"] == "gpt-4"

    def test_save_request(self, sample_prompt):
        revision = sample_prompt.save_request(
            prompt="Input prompt",
            response_text="Generated response",
            model="gpt-4",
            prompt_tokens=10,
            response_tokens=20,
            request_time=int(datetime.now().timestamp()),
            request_id="test_123",
            author="test_author",
            comments="Test save",
        )

        assert revision.content == "Generated response"
        assert revision.request_object.prompt == "Input prompt"
        assert revision.request_object.model == "gpt-4"
        assert revision.author == "test_author"
        assert revision.comments == "Test save"

    def test_get_latest_request(self, sample_prompt, sample_request_object):
        # Test with no revisions
        assert sample_prompt.get_latest_request() is None

        # Add revision with request
        sample_prompt.add_revision(
            content="Test content",
            author="test_author",
            request_object=sample_request_object,
        )

        latest_request = sample_prompt.get_latest_request()
        assert latest_request == sample_request_object
