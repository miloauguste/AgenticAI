"""
Text processor for handling chat logs and internal notes
"""
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

class TextProcessor:
    """Process text files containing chat logs and internal notes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}',
            r'\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}',
            r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]',
            r'\d{2}:\d{2}:\d{2}',
            r'\d{1,2}:\d{2}\s*(AM|PM)'
        ]
        
        self.speaker_patterns = [
            r'^([A-Za-z\s]+):\s*(.+)',
            r'^\[([A-Za-z\s]+)\]\s*(.+)',
            r'^<([A-Za-z\s]+)>\s*(.+)',
            r'^([A-Za-z]+\d*):\s*(.+)',
            r'^Agent\s*(\d+):\s*(.+)',
            r'^Customer:\s*(.+)',
            r'^Support:\s*(.+)'
        ]
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a text file containing chat logs or internal notes
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dict containing processed data and metadata
        """
        try:
            # Read file with encoding detection
            content = self._read_text_file(file_path)
            
            # Detect file type (chat log, notes, etc.)
            file_type = self._detect_file_type(content)
            
            # Process based on file type
            if file_type == 'chat_log':
                processed_data = self._process_chat_log(content)
            elif file_type == 'structured_notes':
                processed_data = self._process_structured_notes(content)
            else:
                processed_data = self._process_plain_text(content)
            
            # Extract conversations/sessions
            conversations = self._extract_conversations(processed_data['messages'])
            
            # Generate summary
            summary = self._generate_summary(conversations, processed_data)
            
            return {
                'file_type': file_type,
                'conversations': conversations,
                'messages': processed_data['messages'],
                'summary': summary,
                'metadata': processed_data['metadata'],
                'processed_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {str(e)}")
            raise
    
    def _read_text_file(self, file_path: str) -> str:
        """Read text file with encoding detection"""
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                self.logger.info(f"Successfully read text file with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Unable to read text file with any of the attempted encodings: {encodings}")
    
    def _detect_file_type(self, content: str) -> str:
        """Detect the type of text file based on content patterns"""
        lines = content.split('\n')
        
        # Check for chat log patterns
        chat_indicators = 0
        structured_indicators = 0
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if not line:
                continue
            
            # Chat log indicators
            if any(re.search(pattern, line) for pattern in self.speaker_patterns):
                chat_indicators += 1
            
            if any(re.search(pattern, line) for pattern in self.timestamp_patterns):
                chat_indicators += 1
            
            # Structured notes indicators
            if line.startswith('#') or line.startswith('##'):
                structured_indicators += 1
            
            if line.startswith('- ') or line.startswith('* '):
                structured_indicators += 1
            
            if re.match(r'^\d+\.', line):
                structured_indicators += 1
        
        if chat_indicators >= 3:
            return 'chat_log'
        elif structured_indicators >= 2:
            return 'structured_notes'
        else:
            return 'plain_text'
    
    def _process_chat_log(self, content: str) -> Dict[str, Any]:
        """Process chat log format text"""
        lines = content.split('\n')
        messages = []
        current_message = None
        line_number = 0
        
        for line in lines:
            line_number += 1
            line = line.strip()
            
            if not line:
                continue
            
            # Try to extract timestamp
            timestamp = self._extract_timestamp(line)
            
            # Try to extract speaker and message
            speaker_match = self._extract_speaker_message(line)
            
            if speaker_match:
                # Save previous message if exists
                if current_message:
                    messages.append(current_message)
                
                speaker, message = speaker_match
                current_message = {
                    'message_id': len(messages) + 1,
                    'speaker': speaker.strip(),
                    'message': message.strip(),
                    'timestamp': timestamp,
                    'line_number': line_number,
                    'word_count': len(message.split()),
                    'char_count': len(message)
                }
            elif current_message:
                # Continue previous message (multi-line)
                current_message['message'] += ' ' + line
                current_message['word_count'] = len(current_message['message'].split())
                current_message['char_count'] = len(current_message['message'])
        
        # Don't forget the last message
        if current_message:
            messages.append(current_message)
        
        return {
            'messages': messages,
            'metadata': {
                'total_lines': line_number,
                'total_messages': len(messages),
                'unique_speakers': len(set(msg['speaker'] for msg in messages if msg['speaker']))
            }
        }
    
    def _process_structured_notes(self, content: str) -> Dict[str, Any]:
        """Process structured notes (markdown-like format)"""
        lines = content.split('\n')
        messages = []
        current_section = None
        line_number = 0
        
        for line in lines:
            line_number += 1
            original_line = line
            line = line.strip()
            
            if not line:
                continue
            
            # Check for headers
            if line.startswith('#'):
                # Save previous section
                if current_section:
                    messages.append(current_section)
                
                header_level = len(line) - len(line.lstrip('#'))
                header_text = line.lstrip('#').strip()
                
                current_section = {
                    'message_id': len(messages) + 1,
                    'type': 'header',
                    'level': header_level,
                    'title': header_text,
                    'content': '',
                    'line_number': line_number,
                    'timestamp': None
                }
            
            elif line.startswith(('- ', '* ')) or re.match(r'^\d+\.', line):
                # List item
                if current_section:
                    if current_section['content']:
                        current_section['content'] += '\n'
                    current_section['content'] += line
                else:
                    messages.append({
                        'message_id': len(messages) + 1,
                        'type': 'list_item',
                        'content': line,
                        'line_number': line_number,
                        'timestamp': self._extract_timestamp(line)
                    })
            
            else:
                # Regular content
                if current_section:
                    if current_section['content']:
                        current_section['content'] += ' '
                    current_section['content'] += line
                else:
                    messages.append({
                        'message_id': len(messages) + 1,
                        'type': 'content',
                        'content': line,
                        'line_number': line_number,
                        'timestamp': self._extract_timestamp(line)
                    })
        
        # Don't forget the last section
        if current_section:
            messages.append(current_section)
        
        # Add word and character counts
        for message in messages:
            content = message.get('content', message.get('title', ''))
            message['word_count'] = len(content.split())
            message['char_count'] = len(content)
        
        return {
            'messages': messages,
            'metadata': {
                'total_lines': line_number,
                'total_sections': len([m for m in messages if m.get('type') == 'header']),
                'total_items': len(messages)
            }
        }
    
    def _process_plain_text(self, content: str) -> Dict[str, Any]:
        """Process plain text format"""
        # Split into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        messages = []
        for i, paragraph in enumerate(paragraphs):
            timestamp = self._extract_timestamp(paragraph)
            
            messages.append({
                'message_id': i + 1,
                'type': 'paragraph',
                'content': paragraph,
                'timestamp': timestamp,
                'word_count': len(paragraph.split()),
                'char_count': len(paragraph)
            })
        
        return {
            'messages': messages,
            'metadata': {
                'total_paragraphs': len(paragraphs),
                'total_characters': len(content)
            }
        }
    
    def _extract_timestamp(self, text: str) -> Optional[str]:
        """Extract timestamp from text"""
        for pattern in self.timestamp_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0).strip('[]')
        return None
    
    def _extract_speaker_message(self, text: str) -> Optional[Tuple[str, str]]:
        """Extract speaker and message from text"""
        for pattern in self.speaker_patterns:
            match = re.match(pattern, text)
            if match:
                if len(match.groups()) == 2:
                    return match.group(1), match.group(2)
                elif len(match.groups()) == 1:
                    return 'Unknown', match.group(1)
        return None
    
    def _extract_conversations(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract conversations/sessions from messages"""
        conversations = []
        current_conversation = None
        conversation_id = 1
        
        for message in messages:
            # Simple heuristic: new conversation if there's a significant time gap
            # or if it's the first message
            start_new_conversation = False
            
            if not current_conversation:
                start_new_conversation = True
            elif message.get('timestamp') and current_conversation.get('last_timestamp'):
                # This is a simplified approach - in practice, you'd want more sophisticated logic
                start_new_conversation = False  # For now, keep all in one conversation
            
            if start_new_conversation:
                if current_conversation:
                    conversations.append(current_conversation)
                
                current_conversation = {
                    'conversation_id': conversation_id,
                    'messages': [message],
                    'start_timestamp': message.get('timestamp'),
                    'last_timestamp': message.get('timestamp'),
                    'participants': set(),
                    'message_count': 1,
                    'total_words': message.get('word_count', 0),
                    'total_chars': message.get('char_count', 0)
                }
                conversation_id += 1
            else:
                current_conversation['messages'].append(message)
                current_conversation['last_timestamp'] = message.get('timestamp') or current_conversation['last_timestamp']
                current_conversation['message_count'] += 1
                current_conversation['total_words'] += message.get('word_count', 0)
                current_conversation['total_chars'] += message.get('char_count', 0)
            
            # Track participants
            if message.get('speaker'):
                current_conversation['participants'].add(message['speaker'])
        
        # Don't forget the last conversation
        if current_conversation:
            conversations.append(current_conversation)
        
        # Convert participants set to list
        for conv in conversations:
            conv['participants'] = list(conv['participants'])
        
        return conversations
    
    def _generate_summary(self, conversations: List[Dict[str, Any]], processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_messages = sum(conv['message_count'] for conv in conversations)
        total_words = sum(conv['total_words'] for conv in conversations)
        total_chars = sum(conv['total_chars'] for conv in conversations)
        
        # Find longest conversation
        longest_conv = max(conversations, key=lambda x: x['message_count']) if conversations else None
        
        # Collect all participants
        all_participants = set()
        for conv in conversations:
            all_participants.update(conv['participants'])
        
        summary = {
            'total_conversations': len(conversations),
            'total_messages': total_messages,
            'total_words': total_words,
            'total_characters': total_chars,
            'average_messages_per_conversation': total_messages / len(conversations) if conversations else 0,
            'average_words_per_message': total_words / total_messages if total_messages else 0,
            'unique_participants': len(all_participants),
            'participants_list': list(all_participants),
            'longest_conversation': {
                'id': longest_conv['conversation_id'],
                'message_count': longest_conv['message_count'],
                'participants': longest_conv['participants']
            } if longest_conv else None,
            'file_metadata': processed_data['metadata']
        }
        
        return summary
    
    def search_messages(self, messages: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Search for messages containing specific content"""
        query = query.lower()
        matching_messages = []
        
        for message in messages:
            content = message.get('content', message.get('message', ''))
            title = message.get('title', '')
            
            if query in content.lower() or query in title.lower():
                message_copy = message.copy()
                message_copy['match_score'] = content.lower().count(query) + title.lower().count(query)
                matching_messages.append(message_copy)
        
        # Sort by match score (descending)
        matching_messages.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matching_messages
    
    def get_conversation_by_id(self, conversations: List[Dict[str, Any]], conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific conversation by ID"""
        for conv in conversations:
            if conv['conversation_id'] == conversation_id:
                return conv
        return None
    
    def export_conversations(self, conversations: List[Dict[str, Any]], output_path: str) -> None:
        """Export conversations to a structured text file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("PROCESSED CONVERSATIONS\n")
                f.write("=" * 50 + "\n\n")
                
                for conv in conversations:
                    f.write(f"Conversation ID: {conv['conversation_id']}\n")
                    f.write(f"Participants: {', '.join(conv['participants'])}\n")
                    f.write(f"Messages: {conv['message_count']}\n")
                    if conv['start_timestamp']:
                        f.write(f"Start Time: {conv['start_timestamp']}\n")
                    f.write("-" * 30 + "\n")
                    
                    for message in conv['messages']:
                        speaker = message.get('speaker', 'System')
                        content = message.get('content', message.get('message', ''))
                        timestamp = message.get('timestamp', '')
                        
                        if timestamp:
                            f.write(f"[{timestamp}] ")
                        f.write(f"{speaker}: {content}\n")
                    
                    f.write("\n" + "=" * 50 + "\n\n")
            
            self.logger.info(f"Conversations exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting conversations: {str(e)}")
            raise