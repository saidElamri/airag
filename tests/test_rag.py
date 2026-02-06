from unittest.mock import patch, MagicMock
import rag
import os

@patch('rag.get_vectorstore')
def test_get_rag_chain_initialization(mock_get_vs):
    # Mock vectorstore to avoid needing a real DB in CI
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = MagicMock()
    mock_get_vs.return_value = mock_vs
    
    # Ensure LLM returns something even without API keys (it uses FakeListLLM)
    chain = rag.get_rag_chain()
    assert chain is not None
    assert hasattr(chain, 'invoke') or hasattr(chain, '__call__')

def test_llm_fallback_logic():
    # If no keys are set, it should return FakeListLLM
    with patch.dict(os.environ, {}, clear=True):
        llm = rag.get_llm()
        from langchain_community.llms import FakeListLLM
        # Note: Chat models vs LLM models categorization might vary
        # but in the code it returns FakeListLLM
        assert isinstance(llm, FakeListLLM)
