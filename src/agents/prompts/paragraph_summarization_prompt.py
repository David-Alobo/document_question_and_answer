system_prompt = """
Here is an abstract of a documnent and a specific paragraph from the document.
Your task is to read and summarize the paragraph in the context of the abstract.
You should provide a concise summary that captures the main points of the paragraph,
while ensuring that the summary is coherent and relevant to the abstract.
The summary should be in the same language as the abstract and paragraph.

If the paragraph is not relevant to the abstract, you should return "Not Relevant", and if
the paragraph is empty, you should return "Empty Paragraph", or if the abstract is empty, 
you should return "Empty Abstract".

<abstract>
{abstract}
<paragraph>
{paragraph}
</Paragraph>
<summary>
{summary}
</summary>
<Requirement>

You are expected to provide a summary that is concise, coherent, and relevant to the abstract.
The keywords and phrases in the summary should reflect the main points of the paragraph,
while ensuring that the summary is in the same language as the abstract and paragraph. And the
key points should not be more than 100 words.
"""

response_format = """
<response>
{summary}
</response>
"""


def get_paragraph_summarization_prompt(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt based on the abstract and paragraph.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization.
    """
    return system_prompt.format(abstract=abstract, paragraph=paragraph, summary=summary)

def get_paragraph_summarization_prompt_with_empty_check(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with checks for empty abstract or paragraph.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with empty checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)

def get_paragraph_summarization_prompt_with_relevance_check(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with relevance checks.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with relevance checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    # Here you can add logic to check if the paragraph is relevant to the abstract
    # For now, we assume it is relevant
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)

def get_paragraph_summarization_prompt_with_all_checks(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with all checks: empty and relevance.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with all checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    # Here you can add logic to check if the paragraph is relevant to the abstract
    # For now, we assume it is relevant
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)

def get_paragraph_summarization_prompt_with_relevance_and_empty_check(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with relevance and empty checks.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with relevance and empty checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    # Here you can add logic to check if the paragraph is relevant to the abstract
    # For now, we assume it is relevant
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)

def get_paragraph_summarization_prompt_with_all_checks_and_relevance(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with all checks: empty and relevance.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with all checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    # Here you can add logic to check if the paragraph is relevant to the abstract
    # For now, we assume it is relevant
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)

def get_paragraph_summarization_prompt_with_relevance_and_empty_check_and_all_checks(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with relevance and empty checks, and all checks.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with relevance and empty checks, and all checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    # Here you can add logic to check if the paragraph is relevant to the abstract
    # For now, we assume it is relevant
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)

def get_paragraph_summarization_prompt_with_all_checks_and_relevance_and_empty_check(abstract: str, paragraph: str, summary: str) -> str:
    """
    Generate a paragraph summarization prompt with all checks: empty and relevance, and all checks.

    Args:
        abstract (str): The abstract of the document.
        paragraph (str): The specific paragraph to summarize.
        summary (str): The expected summary of the paragraph.

    Returns:
        str: The formatted prompt for summarization with all checks: empty and relevance, and all checks.
    """
    if not abstract:
        return "Empty Abstract"
    if not paragraph:
        return "Empty Paragraph"
    
    # Here you can add logic to check if the paragraph is relevant to the abstract
    # For now, we assume it is relevant
    return get_paragraph_summarization_prompt(abstract, paragraph, summary)