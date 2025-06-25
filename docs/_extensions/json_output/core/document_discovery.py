"""Document discovery and filtering functionality."""

from typing import List
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment

from ..utils import get_setting


class DocumentDiscovery:
    """Handles document discovery, filtering, and hierarchical relationships."""
    
    def __init__(self, app: Sphinx, json_builder):
        self.app = app
        self.env = app.env
        self.config = app.config
        self.json_builder = json_builder  # Reference to main builder for metadata access
    
    def get_child_documents(self, parent_docname: str) -> List[str]:
        """Get all child documents for a parent directory."""
        if parent_docname == 'index':
            parent_path = ''
        elif parent_docname.endswith('/index'):
            parent_path = parent_docname[:-6]  # Remove '/index'
        else:
            # Not a directory index, no children
            return []
        
        children = []
        for docname in self.env.all_docs:
            if self.is_hidden_document(docname):
                continue
                
            # Skip the parent itself
            if docname == parent_docname:
                continue
                
            # Check if this document is a child of the parent
            if parent_path == '':
                # Root level - include all docs
                children.append(docname)
            elif docname.startswith(parent_path + '/'):
                children.append(docname)
                
        return sorted(children)
    
    def is_hidden_document(self, docname: str) -> bool:
        """Check if a document should be considered hidden."""
        # Skip documents that match exclude patterns
        for pattern in get_setting(self.config, 'exclude_patterns', []):
            if docname.startswith(pattern):
                return True
                
        # Skip documents with 'hidden' or 'draft' in metadata
        metadata = self.json_builder.extract_document_metadata(docname)
        if metadata.get('hidden') or metadata.get('draft'):
            return True
            
        # Skip documents that wouldn't generate JSON
        if not self.json_builder.should_generate_json(docname):
            return True
            
        return False
    
    def get_all_documents_recursive(self) -> List[str]:
        """Get all non-hidden documents recursively."""
        all_docs = []
        for docname in self.env.all_docs:
            if not self.is_hidden_document(docname):
                all_docs.append(docname)
        return sorted(all_docs)
    
    def get_section_path(self, docname: str) -> List[str]:
        """Get hierarchical section path for navigation."""
        parts = docname.split('/')
        
        # Filter out common file names to get clean section path
        filtered_parts = []
        for part in parts:
            if part not in ['index', 'README']:
                filtered_parts.append(part.replace('-', ' ').replace('_', ' ').title())
        
        return filtered_parts
    
    def detect_document_type(self, docname: str, title: str, content: str) -> str:
        """Detect document type for better search categorization."""
        docname_lower = docname.lower()
        title_lower = title.lower()
        content_lower = content.lower()[:1000]  # First 1000 chars
        
        # Check for common document types
        if 'tutorial' in docname_lower or 'tutorial' in title_lower:
            return 'tutorial'
        elif 'guide' in docname_lower or 'guide' in title_lower:
            return 'guide'
        elif 'reference' in docname_lower or 'api' in docname_lower:
            return 'reference'
        elif 'example' in docname_lower or 'examples' in docname_lower:
            return 'example'
        elif 'troubleshoot' in docname_lower or 'faq' in docname_lower:
            return 'troubleshooting'
        elif 'install' in docname_lower or 'setup' in docname_lower:
            return 'installation'
        elif docname.endswith('/index'):
            return 'overview'
        elif any(word in content_lower for word in ['$ ', 'pip install', 'docker run', 'git clone']):
            return 'tutorial'
        elif any(word in content_lower for word in ['class ', 'def ', 'function', 'method', 'parameter']):
            return 'reference'
        else:
            return 'documentation' 