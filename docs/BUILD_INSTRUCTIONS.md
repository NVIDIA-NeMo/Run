# Documentation Build Instructions

Complete guide for building the nemo-run documentation.

## **Prerequisites and Requirements**

### **1. System Requirements**

- **Python** (version specified in `.python-version`)
- **uv** package manager (fast Python package installer)
- **Windows PowerShell** (for Windows users)

### **2. Required Dependencies** (from `requirements-docs.txt`)

```
sphinx
myst-parser
sphinx-autodoc2
sphinx-copybutton
nvidia-sphinx-theme
sphinx-autobuild
sphinx-design
pinecone
openai
python-dotenv
sphinxcontrib-mermaid
swagger-plugin-for-sphinx
```

## **Setup Steps**

### **Step 1: Install uv (if not already installed)**

```bash
# Windows PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

### **Step 2: Set up the documentation environment**

```bash
# From project root directory
make docs-env
```

This command will:

- Check if `uv` is installed
- Create virtual environment `.venv-docs`
- Install all dependencies from `requirements-docs.txt`

## **Build Commands**

### **Basic Build Commands**

```bash
# Navigate to docs directory
cd docs

# Build HTML documentation
uv run --active python -m sphinx -b html . _build/html
```

### **Alternative Build Commands**

```bash
# Using Makefile (from project root)
make docs-html

# Strict build (fails on warnings)
make docs-publish

# With environment tags
uv run --active python -m sphinx -b html -t internal . _build/html
uv run --active python -m sphinx -b html -t ga . _build/html
uv run --active python -m sphinx -b html -t ea . _build/html
uv run --active python -m sphinx -b html -t draft . _build/html
```

### **Development Commands**

```bash
# Start live-reload server
make docs-live

# Clean built documentation
make docs-clean

# Clean and rebuild
make docs-clean && make docs-html
```

## **Complete Setup and Build Process**

### **One-time Setup:**

```bash
# 1. Install uv (if needed)
pip install uv

# 2. Set up environment
make docs-env
```

### **Regular Build Process:**

```bash
# Option 1: Using Makefile (recommended)
make docs-html

# Option 2: Direct Sphinx command
cd docs
uv run --active python -m sphinx -b html . _build/html
```

### **Development Workflow:**

```bash
# Start live server for development
make docs-live

# In another terminal, edit documentation files
# Changes will automatically rebuild and refresh browser
```

## **Output Location**

After successful build:

- **HTML files**: `docs/_build/html/`
- **Main index**: `docs/_build/html/nemo-run-index.html`
- **Search page**: `docs/_build/html/search.html`

## **Troubleshooting**

### **If uv is not found:**

```bash
# Restart terminal after installation
# Or manually add uv to PATH
```

### **If virtual environment issues:**

```bash
# Recreate environment
rm -rf .venv-docs
make docs-env
```

### **If build fails:**

```bash
# Clean and rebuild
make docs-clean
make docs-html
```

## **Environment-Specific Builds**

```bash
# Internal use
make docs-html-internal

# General Availability
make docs-html-ga

# Early Access
make docs-html-ea

# Draft
make docs-html-draft
```

## **All Available Makefile Commands**

### **Basic Commands**

```bash
docs-html          # Build HTML documentation
docs-publish       # Build HTML documentation for publication (fail on warnings)
docs-clean         # Clean built documentation
docs-live          # Start live-reload server (sphinx-autobuild)
docs-env           # Set up docs virtual environment with uv
```

### **Environment-Specific Commands**

```bash
# Internal environment builds
docs-html-internal
docs-publish-internal
docs-live-internal

# GA (General Availability) environment builds
docs-html-ga
docs-publish-ga
docs-live-ga

# EA (Early Access) environment builds
docs-html-ea
docs-publish-ea
docs-live-ea

# Draft environment builds
docs-html-draft
docs-publish-draft
docs-live-draft
```

### **Pinecone Integration Commands**

```bash
docs-pinecone-test        # Test Pinecone connection
docs-pinecone-upload-dry  # Upload documentation to Pinecone (dry run)
docs-pinecone-upload      # Upload documentation to Pinecone
docs-pinecone-update      # Build docs and update Pinecone index
```

## **Cross-Platform Compatibility**

The Makefile automatically detects your OS and uses the appropriate commands:

- **Windows**: Uses `.venv-docs\Scripts\` paths
- **Unix/Linux/macOS**: Uses `.venv-docs/bin/` paths

## **Usage Examples**

```bash
# Quick build for development
make docs-html

# Production build (strict)
make docs-publish

# Development with live reload
make docs-live

# Build with specific environment tag
make docs-html DOCS_ENV=ga

# Clean and rebuild
make docs-clean && make docs-html
```

## **Sphinx Command Reference**

### **Basic Sphinx Command**

```bash
uv run --active python -m sphinx -b html . _build/html
```

### **Command Breakdown:**

- `uv run --active` - Uses the active virtual environment (`.venv-docs`)
- `python -m sphinx` - Runs Sphinx as a Python module
- `-b html` - Specifies the HTML builder
- `.` - Source directory (current directory)
- `_build/html` - Output directory for the built HTML files

### **Strict Build (Fails on Warnings):**

```bash
uv run --active python -m sphinx --fail-on-warning --builder html . _build/html
```

The documentation build system is designed to be cross-platform and handles Windows PowerShell automatically through the Makefile configuration.

## **Setting Up Documentation in a New Repository**

### **Copying docs-example-project-setup to Your GitHub Repo**

If you want to set up documentation in a new GitHub repository using the NVIDIA docs-example-project-setup as a template:

#### **Step 1: Clone and Fork the Example Project**
```bash
# Clone the NVIDIA docs example project
git clone https://gitlab-master.nvidia.com/llane/docs-example-project-setup.git

# Navigate to the cloned directory
cd docs-example-project-setup

# Create a new branch for staging/sandbox testing
git checkout -b staging
```

#### **Step 1.5: Archive Source Repository Files**
```bash
# From your new GitHub repository root
# Create an archive directory to store the original source files
mkdir archive

# Copy all files from the source repository to archive
cp -r docs-example-project-setup/* ./archive/

# Or if you want to preserve the git history in archive
cd archive
git clone https://gitlab-master.nvidia.com/llane/docs-example-project-setup.git
cd ..

# This archive directory serves as your starting point reference
# You can always go back to see the original structure and content
```

#### **Step 2: Copy Documentation Files to Your New Repo**
```bash
# From your new GitHub repository root
# Copy the essential documentation files and directories:

# Copy the docs directory structure
cp -r docs-example-project-setup/docs/ ./docs/

# Copy the Makefile (contains docs build targets)
cp docs-example-project-setup/Makefile ./

# Copy requirements file
cp docs-example-project-setup/requirements-docs.txt ./

# Copy Sphinx configuration
cp docs-example-project-setup/docs/conf.py ./docs/

# Copy any custom extensions
cp -r docs-example-project-setup/docs/_extensions/ ./docs/_extensions/
```

#### **Step 3: Customize for Your Project**
```bash
# Edit the Sphinx configuration
# Update docs/conf.py with your project details:
# - project name
# - version
# - author
# - theme settings
# - extensions

# Update the Makefile if needed for your project structure

# Modify documentation content in docs/ directory
# - Update index files
# - Customize content for your project
# - Add your own documentation pages
```

#### **Step 4: Set Up the Documentation Environment**
```bash
# Install uv (if not already installed)
pip install uv

# Set up the documentation environment
make docs-env

# Build the documentation
make docs-html
```

#### **Step 5: Test and Iterate**
```bash
# Start live server for development
make docs-live

# Make changes to documentation files
# Changes will automatically rebuild and refresh browser

# Test different build environments
make docs-html-internal
make docs-html-ga
make docs-html-ea
make docs-html-draft
```

#### **Step 6: Commit and Push to Your Repository**
```bash
# Add all documentation files
git add .

# Commit the changes
git commit -m "Add documentation setup from NVIDIA docs-example-project-setup"

# Push to your staging branch
git push origin staging
```

### **Important Notes for Sandbox Testing**

- **Environment Variables**: You may need to set up environment variables for features like Pinecone integration
- **Custom Extensions**: Some extensions may require additional configuration or API keys
- **Theme Customization**: The NVIDIA theme can be customized for your project branding
- **Content Structure**: Modify the documentation structure to match your project's needs
- **Build Testing**: Test all build environments (internal, ga, ea, draft) to ensure they work correctly

### **Troubleshooting New Setup**

```bash
# If build fails due to missing dependencies
make docs-clean
make docs-env
make docs-html

# If extensions don't work
# Check docs/conf.py for proper extension configuration
# Verify all required packages are in requirements-docs.txt

# If theme issues occur
# Check theme configuration in docs/conf.py
# Verify nvidia-sphinx-theme is properly installed
```
