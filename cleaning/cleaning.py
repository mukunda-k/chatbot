import os
import re
from pathlib import Path
from bs4 import BeautifulSoup, Comment, NavigableString
from typing import List, Optional
import unicodedata

class HTMLCleaner:
    """HTML cleaner that extracts clean text while preserving links and list structure"""
    
    def __init__(self):
        self.remove_tags = {
            'script', 'style', 'noscript', 'iframe', 'object', 'embed',
            'applet', 'form', 'input', 'textarea', 'button', 'select',
            'option', 'meta', 'link', 'base', 'svg', 'canvas'
        }
        
        self.nav_tags = {
            'nav', 'header', 'footer', 'aside', 'menu', 'menuitem'
        }
        
        self.remove_patterns = [
            r'nav', r'menu', r'header', r'footer', r'sidebar', r'ads?',
            r'advertisement', r'cookie', r'popup', r'modal', r'overlay',
            r'banner', r'breadcrumb', r'pagination', r'search', r'filter',
            r'social', r'share', r'comment', r'related', r'recommend',
            r'subscribe', r'newsletter', r'promo', r'offer', r'discount',
            r'flight', r'lounge', r'terminal', r'guide', r'direction',
            r'follow', r'sign', r'app', r'download', r'policy', r'condition'
        ]
    
    def clean_folder(self, folder_path: str, output_folder: Optional[str] = None, file_pattern: str = "*.html") -> None:
        """Clean all HTML files in a folder and save as text files"""
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        if output_folder is None:
            script_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()
            output_folder = script_dir.parent / f"{folder_path.name}_cleaned"
        else:
            output_folder = Path(output_folder)
        
        output_folder.mkdir(exist_ok=True)
        
        html_files = list(folder_path.rglob(file_pattern))
        if not html_files:
            print(f"No HTML files found matching pattern: {file_pattern}")
            return
        
        print(f"Found {len(html_files)} HTML files to clean...")
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                cleaned_text = self.clean_html(html_content)
                
                if not cleaned_text.strip():
                    print(f"No content extracted from {html_file.name}")
                    continue
                
                relative_path = html_file.relative_to(folder_path)
                output_file_dir = output_folder / relative_path.parent
                output_file_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_file_dir / f"{html_file.stem}.txt"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                print(f"Processed: {html_file.name}")
                
            except Exception as e:
                print(f"Error processing {html_file.name}: {e}")
        
        print(f"Cleaning complete! Results saved in: {output_folder}")
    
    def clean_html(self, html_content: str) -> str:
        """Clean HTML and return plain text with preserved links and list structure"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        self._remove_comments(soup)
        self._remove_unwanted_tags(soup)
        self._remove_by_patterns(soup)
        self._remove_empty_elements(soup)
        
        text = self._extract_text(soup)
        return self._post_process_text(text)
    
    def _remove_comments(self, soup: BeautifulSoup) -> None:
        """Remove HTML comments"""
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
    
    def _remove_unwanted_tags(self, soup: BeautifulSoup) -> None:
        """Remove unwanted tags and their content"""
        for tag_name in self.remove_tags | self.nav_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
    
    def _remove_by_patterns(self, soup: BeautifulSoup) -> None:
        """Remove elements by class/id patterns and common navigation text"""
        for pattern in self.remove_patterns:
            for tag in soup.find_all(class_=re.compile(pattern, re.I)):
                tag.decompose()
            for tag in soup.find_all(id=re.compile(pattern, re.I)):
                tag.decompose()
        
        # Remove elements with navigation-like text content
        nav_keywords = ['search', 'popular searches', 'lounges', 'happening now', 'view now', 
                       'great deals', 'follow us', 'sign up', 'download', 'privacy policy',
                       'flight information', 'arrival guide', 'departure guide', 'contact information']
        
        for tag in soup.find_all():
            text = tag.get_text(strip=True).lower()
            if any(keyword in text for keyword in nav_keywords) and len(text) < 100:
                tag.decompose()
    
    def _remove_empty_elements(self, soup: BeautifulSoup) -> None:
        """Remove empty elements"""
        for _ in range(3):
            for tag in soup.find_all():
                if not tag.get_text(strip=True) and not tag.find_all(['img', 'br', 'hr']):
                    tag.decompose()
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract text with preserved structure for links and lists"""
        lines = []
        
        # Try to find main content areas first
        main_content = (soup.find('main') or 
                       soup.find('article') or 
                       soup.find(class_=re.compile(r'content|main|body', re.I)) or
                       soup.find(id=re.compile(r'content|main|body', re.I)) or
                       soup.body or soup)
        
        self._traverse_for_text(main_content, lines)
        return '\n'.join(lines)
    
    def _traverse_for_text(self, element, lines: List[str]) -> None:
        """Recursively traverse elements to extract structured text"""
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text and len(text) > 1:
                lines.append(text)
            return
        
        if not hasattr(element, 'name'):
            return
        
        # Handle line breaks
        if element.name in ['br']:
            lines.append('')
            return
        
        if element.name in ['hr']:
            lines.append('---')
            return
        
        # Handle headings
        if element.name and re.match(r'h[1-6]', element.name):
            text = element.get_text(strip=True)
            if text:
                level = int(element.name[1])
                lines.append(f"{'#' * level} {text}")
                lines.append('')
            return
        
        # Handle links - convert to text:url format
        if element.name == 'a':
            text = element.get_text(strip=True)
            href = element.get('href', '')
            if text and href:
                lines.append(f"{text}:{href}")
            elif text:
                lines.append(text)
            return
        
        # Handle list items
        if element.name == 'li':
            text = element.get_text(strip=True)
            if text:
                lines.append(f"• {text}")
            return
        
        # Handle lists - add spacing
        if element.name in ['ul', 'ol']:
            for child in element.children:
                self._traverse_for_text(child, lines)
            lines.append('')
            return
        
        # Handle paragraphs and divs - filter out short navigation text
        if element.name in ['p', 'div', 'section', 'article']:
            # Process children to handle nested links
            child_lines = []
            for child in element.children:
                if isinstance(child, NavigableString):
                    text = str(child).strip()
                    if text:
                        child_lines.append(text)
                elif hasattr(child, 'name') and child.name == 'a':
                    text = child.get_text(strip=True)
                    href = child.get('href', '')
                    if text and href:
                        child_lines.append(f"{text}:{href}")
                    elif text:
                        child_lines.append(text)
                else:
                    # For other nested elements, just get text
                    text = child.get_text(strip=True) if hasattr(child, 'get_text') else str(child).strip()
                    if text:
                        child_lines.append(text)
            
            if child_lines:
                combined_text = ' '.join(child_lines)
                # Filter out navigation-like content
                nav_indicators = ['view now', 'sign up', 'follow us', 'download', 'contact', 
                                'privacy policy', 'terms', 'conditions', '© 2025']
                
                if (len(combined_text) > 20 and 
                    not any(indicator in combined_text.lower() for indicator in nav_indicators)):
                    lines.append(combined_text)
                    lines.append('')
            return
        
        # For other elements, traverse children
        if hasattr(element, 'children'):
            for child in element.children:
                self._traverse_for_text(child, lines)
    
    def _post_process_text(self, text: str) -> str:
        """Clean up the extracted text"""
        text = unicodedata.normalize('NFKD', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                filtered_lines.append('')
            elif len(line) > 2 or line.startswith('#') or line.startswith('•'):
                filtered_lines.append(line)
        
        # Remove excessive empty lines
        result_lines = []
        empty_count = 0
        
        for line in filtered_lines:
            if not line:
                empty_count += 1
                if empty_count <= 1:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines).strip()

def clean_html_for_text(input_folder: str, output_folder: str = None) -> None:
    """Clean HTML files and convert to plain text with preserved links and lists"""
    cleaner = HTMLCleaner()
    cleaner.clean_folder(input_folder, output_folder)

if __name__ == "__main__":
    cleaner = HTMLCleaner()
    
    __location__ = os.path.dirname(os.path.abspath(__file__))
    __input__ = os.path.join(os.path.dirname(__location__), "data")
    __output__ = os.path.join(os.path.dirname(__location__), "cleaned_data")
    os.makedirs(__output__, exist_ok=True)
    
    cleaner.clean_folder(__input__, __output__)