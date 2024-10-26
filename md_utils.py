import re

MAX_MESSAGE_LENGTH = 4000

def clean_and_split(text: str) -> list[str]:
    """
    Clean the Markdown text and split it into chunks that fit Telegram's message size limit.
    """
    cleaned_text = clean_markdown(text)
    return split_text(cleaned_text)

def clean_markdown(text: str) -> str:
    """
    Clean Markdown to ensure all formatting is properly closed using regex.
    """
    # Define regex patterns for Markdown elements
    patterns = [
        (r'\*([^*\n]+)(?<!\*)\*?', r'*\1*'),  # Bold
        (r'_([^_\n]+)(?<!_)_?', r'_\1_'),  # Italic
        (r'~([^~\n]+)(?<!~)~?', r'~\1~'),  # Strikethrough
        (r'\|([^|\n]+)(?<!\|)\|?', r'|\1|'),  # Spoiler
        (r'`([^`\n]+)(?<!`)`?', r'`\1`'),  # Inline code
        (r'```([\s\S]*?)```', r'```\1```')  # Code block
    ]

    # Apply each pattern
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)

    # Remove any remaining single Markdown characters at the start of lines
    text = re.sub(r'^([*_~|`])\s', r'\1', text, flags=re.MULTILINE)

    return text

def split_text(text: str) -> list[str]:
    """
    Split text into chunks that fit within Telegram's message size limit.
    """
    chunks = []
    current_chunk = ""

    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line + '\n'
        else:
            current_chunk += line + '\n'

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram's MarkdownV2.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)