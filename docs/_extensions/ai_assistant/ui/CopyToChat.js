(function () {
  function ready(fn) {
    if (document.readyState !== 'loading') { fn(); } else { document.addEventListener('DOMContentLoaded', fn); }
  }

  function getLanguage(el) {
    const c = (el && el.getAttribute && (el.getAttribute('class') || '')) + ' ' + ((el && el.closest && el.closest('.highlight') && el.closest('.highlight').getAttribute('class')) || '');
    let m = c && c.match(/language-([\w-]+)/i); if (m) return m[1];
    m = c && c.match(/highlight-([\w-]+)/i); if (m) return m[1];
    return '';
  }

  function getSelectionWithin(el) {
    const sel = window.getSelection && window.getSelection();
    if (!sel || sel.rangeCount === 0) return '';
    const range = sel.getRangeAt(0);
    if (!el || !el.contains || !el.contains(range.commonAncestorContainer)) return '';
    return sel.toString() || '';
  }

  function truncate(text, max = 8000) {
    if (!text) return '';
    if (text.length <= max) return text;
    return text.slice(0, max - 20) + "\n...";
  }

  function buildPrompt(lang, code) {
    const lead = 'Explain the following code snippet clearly and concisely. Include purpose, key steps, and potential pitfalls. If relevant, suggest improvements.';
    const fence = '```' + (lang || '') + '\n' + code + '\n```';
    return lead + '\n\n' + fence;
  }

  function showCodeExplanation(container, lang, code, scope) {
    // Remove any existing explanation
    const existing = container.querySelector('.code-explanation');
    if (existing) {
      existing.remove();
      return; // Toggle off if already showing
    }

    // Create explanation panel
    const explanation = document.createElement('div');
    explanation.className = 'code-explanation';
    explanation.innerHTML = `
      <div class="explanation-header">
        <span class="explanation-icon">💡</span>
        <span class="explanation-title">Code Explanation</span>
        <button class="explanation-close" onclick="this.parentElement.parentElement.remove()">×</button>
      </div>
      <div class="explanation-content">
        <div class="explanation-loading">
          <div class="loading-spinner"></div>
          <span>Analyzing ${scope === 'selection' ? 'selected' : 'code'}...</span>
        </div>
      </div>
    `;

    // Add explanation styles if not already added
    if (!document.getElementById('code-explanation-styles')) {
      const style = document.createElement('style');
      style.id = 'code-explanation-styles';
      style.textContent = `
        .code-explanation {
          margin-top: 12px;
          border: 1px solid #e1e5e9;
          border-radius: 8px;
          background: #f8f9fa;
          font-family: inherit;
          font-size: 14px;
          max-height: 400px;
          overflow-y: auto;
        }

        .explanation-header {
          display: flex;
          align-items: center;
          padding: 8px 12px;
          background: #e9ecef;
          border-bottom: 1px solid #e1e5e9;
          border-radius: 8px 8px 0 0;
          font-weight: 500;
        }

        .explanation-icon {
          margin-right: 6px;
        }

        .explanation-title {
          flex-grow: 1;
          color: #333;
        }

        .explanation-close {
          background: none;
          border: none;
          font-size: 18px;
          cursor: pointer;
          padding: 0;
          color: #666;
          margin-left: 8px;
        }

        .explanation-close:hover {
          color: #333;
        }

        .explanation-content {
          padding: 12px;
          line-height: 1.5;
        }

        .explanation-loading {
          display: flex;
          align-items: center;
          color: #666;
          font-style: italic;
        }

        .loading-spinner {
          width: 16px;
          height: 16px;
          border: 2px solid #e9ecef;
          border-top: 2px solid #6c757d;
          border-radius: 50%;
          animation: spin 1s linear infinite;
          margin-right: 8px;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }

        .explanation-section {
          margin-bottom: 12px;
        }

        .explanation-section h4 {
          margin: 0 0 4px 0;
          color: #495057;
          font-size: 13px;
          font-weight: 600;
        }

        .explanation-section p {
          margin: 0;
          color: #6c757d;
        }

        .explanation-section ul {
          margin: 4px 0;
          padding-left: 16px;
          color: #6c757d;
        }

        .explanation-section li {
          margin-bottom: 2px;
        }
      `;
      document.head.appendChild(style);
    }

    container.appendChild(explanation);

    // Generate the explanation
    generateCodeExplanation(lang, code, scope).then(result => {
      const contentDiv = explanation.querySelector('.explanation-content');
      contentDiv.innerHTML = result;
    }).catch(() => {
      const contentDiv = explanation.querySelector('.explanation-content');
      contentDiv.innerHTML = `
        <div class="explanation-section">
          <p style="color: #dc3545;">Unable to generate explanation. Try the "Ask AI" button for detailed analysis.</p>
        </div>
      `;
    });
  }

  function generateCodeExplanation(lang, code, scope) {
    return new Promise((resolve) => {
      // Simulate processing time for better UX
      setTimeout(() => {
        const explanation = analyzeCode(lang, code, scope);
        resolve(explanation);
      }, 800);
    });
  }

  function analyzeCode(lang, code, scope) {
    // Clean and analyze the code
    const trimmedCode = code.trim();
    const lines = trimmedCode.split('\n');

    // Detect NeMo Run patterns
    const isNemoRun = /run\.Config|run\.Partial|nemo_run/i.test(code);
    const isPython = lang === 'python' || /^(import|from|def|class|if __name__)/.test(trimmedCode);
    const isConfig = /\.yaml|\.yml|config:|parameters:/i.test(code);
    const isBash = lang === 'bash' || /^(#!\/bin\/bash|export|cd |mkdir|pip install)/m.test(code);

    let sections = [];

    // Purpose section
    if (isNemoRun) {
      if (/run\.Config/.test(code)) {
        sections.push({
          title: 'Purpose',
          content: 'This creates a NeMo Run configuration object that defines how to execute a task with specific parameters. It allows for flexible, serializable job definitions.'
        });
      } else if (/run\.Partial/.test(code)) {
        sections.push({
          title: 'Purpose',
          content: 'This uses NeMo Run\'s Partial to create a partially configured function with some arguments pre-filled, enabling flexible job composition.'
        });
      } else {
        sections.push({
          title: 'Purpose',
          content: 'This code is part of a NeMo Run workflow for distributed machine learning job execution.'
        });
      }
    } else if (isPython && /class/.test(code)) {
      sections.push({
        title: 'Purpose',
        content: 'This Python class definition creates a reusable component with methods and attributes.'
      });
    } else if (isPython && /def/.test(code)) {
      sections.push({
        title: 'Purpose',
        content: 'This Python function performs a specific task and can be called with arguments.'
      });
    } else if (isConfig) {
      sections.push({
        title: 'Purpose',
        content: 'This configuration file defines parameters and settings for the application.'
      });
    } else if (isBash) {
      sections.push({
        title: 'Purpose',
        content: 'This shell script automates command-line operations and environment setup.'
      });
    }

    // Key components
    const keyPoints = [];

    if (isNemoRun) {
      if (/executor/.test(code)) keyPoints.push('Defines execution environment (local, SLURM, etc.)');
      if (/nodes?=/.test(code)) keyPoints.push('Specifies number of compute nodes');
      if (/devices=/.test(code)) keyPoints.push('Sets GPU/device allocation');
      if (/env_vars/.test(code)) keyPoints.push('Configures environment variables');
    }

    if (isPython) {
      if (/import/.test(code)) keyPoints.push('Imports required modules and dependencies');
      if (/def __init__/.test(code)) keyPoints.push('Constructor method initializes object state');
      if (/@/.test(code)) keyPoints.push('Uses decorators to modify function behavior');
      if (/try:|except:/.test(code)) keyPoints.push('Includes error handling');
    }

    if (isConfig) {
      if (/trainer:/.test(code)) keyPoints.push('Configures training parameters');
      if (/model:/.test(code)) keyPoints.push('Defines model architecture settings');
      if (/data:/.test(code)) keyPoints.push('Specifies data loading configuration');
    }

    if (keyPoints.length > 0) {
      sections.push({
        title: 'Key Components',
        content: `<ul>${keyPoints.map(point => `<li>${point}</li>`).join('')}</ul>`
      });
    }

    // Common patterns/best practices
    const tips = [];

    if (isNemoRun) {
      if (/run\.Config/.test(code)) {
        tips.push('Config objects are serializable and can be saved/loaded');
        tips.push('Use type hints for better validation and IDE support');
      }
      if (lines.length > 10) {
        tips.push('For complex configs, consider breaking into smaller, reusable components');
      }
    }

    if (isPython && /def/.test(code)) {
      if (!/"""/.test(code)) tips.push('Consider adding docstrings for better documentation');
      if (!/typing/.test(code) && /:/.test(code)) tips.push('Type hints improve code clarity and catch errors');
    }

    if (tips.length > 0) {
      sections.push({
        title: 'Best Practices',
        content: `<ul>${tips.map(tip => `<li>${tip}</li>`).join('')}</ul>`
      });
    }

    // Build the HTML
    let html = '';
    sections.forEach(section => {
      html += `
        <div class="explanation-section">
          <h4>${section.title}</h4>
          ${section.content.startsWith('<') ? section.content : `<p>${section.content}</p>`}
        </div>
      `;
    });

    // Add scope info
    if (scope === 'selection') {
      html += `
        <div class="explanation-section">
          <p style="font-size: 12px; color: #868e96; font-style: italic;">
            💡 This explains your selected code. Click "Ask AI" for detailed analysis of the full snippet.
          </p>
        </div>
      `;
    }

    return html || `
      <div class="explanation-section">
        <p>This appears to be ${lang || 'code'} that performs specific operations. For detailed analysis, try the "Ask AI" button.</p>
      </div>
    `;
  }

  function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    const ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } finally { document.body.removeChild(ta); }
    return Promise.resolve();
  }

  function openChat() {
    const url = 'https://chatgpt.com/';
    window.open(url, '_blank', 'noopener');
  }


  function buildToolbar(container, codeEl) {
    // Install toolbar CSS once
    if (!document.getElementById('ai-toolbar-style')) {
      const style = document.createElement('style');
      style.id = 'ai-toolbar-style';
      style.textContent = `
        .ai-toolbar{position:absolute;top:6px;right:6px;display:flex;gap:10px;z-index:2}
        .ai-btn{display:inline-flex;align-items:center;gap:6px;padding:6px 10px;font-size:12px;border:1px solid #d1d5db;border-radius:8px;background:#ffffffcc;color:#111827;cursor:pointer;box-shadow:0 1px 2px rgba(17,24,39,.06)}
        .ai-btn:hover{box-shadow:0 4px 12px rgba(17,24,39,.10)}
        .ai-btn:focus{outline:2px solid rgba(111,176,0,.35);outline-offset:2px}
        .ai-btn__icon{font-size:14px;line-height:1}
        .ai-btn--primary{background:var(--primary,#6FB000);border-color:transparent;color:#fff;box-shadow:0 6px 16px rgba(17,24,39,.12)}
        .ai-btn--primary:hover{filter:brightness(1.05)}
        @media (max-width:480px){.ai-btn__label{display:none}.ai-btn{padding:6px}}
      `;
      document.head.appendChild(style);
    }

    const toolbar = document.createElement('div');
    toolbar.setAttribute('role', 'toolbar');
    toolbar.className = 'ai-toolbar';

    function makeBtn(iconEmoji, label, title, primary) {
      const b = document.createElement('button');
      b.type = 'button'; b.setAttribute('aria-label', label);
      if (title) b.title = title;
      b.className = 'ai-btn' + (primary ? ' ai-btn--primary' : '');
      const icon = document.createElement('span');
      icon.className = 'ai-btn__icon';
      icon.textContent = iconEmoji;
      const text = document.createElement('span');
      text.className = 'ai-btn__label';
      text.textContent = label;
      b.appendChild(icon);
      b.appendChild(text);
      return b;
    }

    // Copy button (raw code or selection)
    const copyBtn = makeBtn('📋', 'Copy', 'Copy code');
    copyBtn.onclick = () => {
      const raw = (codeEl && codeEl.textContent) || '';
      const selected = getSelectionWithin(codeEl);
      const text = (selected && selected.trim()) ? selected : raw;
      copyToClipboard(text);
    };

    // Ask AI button (prompt + code)
    const askBtn = makeBtn('💬', 'Ask AI', 'Copy prompt + code and open chat', true);
    askBtn.onclick = () => {
      const lang = getLanguage(codeEl);
      const raw = (codeEl && codeEl.textContent) || '';
      const selected = getSelectionWithin(codeEl);
      const snippet = truncate((selected && selected.trim()) ? selected : raw);
      const prompt = buildPrompt(lang, snippet);
      copyToClipboard(prompt).then(openChat);
    };

    // Code Explainer button (inline explanation)
    const explainBtn = makeBtn('💡', 'Explain', 'Get instant code explanation');
    explainBtn.onclick = () => {
      const lang = getLanguage(codeEl);
      const raw = (codeEl && codeEl.textContent) || '';
      const selected = getSelectionWithin(codeEl);
      const snippet = (selected && selected.trim()) ? selected : raw;
      showCodeExplanation(container, lang, snippet, selected ? 'selection' : 'full');
    };

    toolbar.appendChild(copyBtn);
    toolbar.appendChild(explainBtn);
    toolbar.appendChild(askBtn);
    container.appendChild(toolbar);
  }

  function installToolbar() {
    // Wrap each code block and inject toolbar
    const highlights = Array.from(document.querySelectorAll('div.highlight'));
    const preOnly = Array.from(document.querySelectorAll('pre:not(.literal-block)')).filter(p => !p.closest('div.highlight'));
    const targets = highlights.concat(preOnly);

    targets.forEach(block => {
      if (block.getAttribute('data-ai-toolbar')) return;
      block.setAttribute('data-ai-toolbar', 'true');

      // Hide default sphinx copy button if present to avoid overlap
      const defaultCopy = block.querySelector('button.copybtn, .copybtn');
      if (defaultCopy) defaultCopy.style.display = 'none';

      // Ensure a wrapper with position: relative
      const wrapper = document.createElement('div');
      wrapper.style.position = 'relative';
      wrapper.style.width = '100%';
      block.parentNode.insertBefore(wrapper, block);
      wrapper.appendChild(block);

      // Add top padding so toolbar never overlaps code
      const pre = block.querySelector('pre') || block;
      if (pre && !pre.style.paddingTop) pre.style.paddingTop = '28px';

      // Determine code element
      const codeEl = block.querySelector('code') || pre;
      buildToolbar(wrapper, codeEl);
    });
  }

  function createPageTranslator() {
    // Only create once per page
    if (document.getElementById('page-translator')) return;

    // Try to find header areas to integrate with
    const headerSelectors = [
      '.navbar',
      '.header',
      '.top-bar',
      '.page-header',
      'header',
      '.bd-header',
      '.site-header',
      '.navbar-nav',
      '.bd-navbar'
    ];

    let headerElement = null;
    for (const selector of headerSelectors) {
      headerElement = document.querySelector(selector);
      if (headerElement) {
        console.log('Found header element:', selector);
        break;
      }
    }

    // Install translator CSS once
    if (!document.getElementById('page-translator-style')) {
      const style = document.createElement('style');
      style.id = 'page-translator-style';

      if (headerElement) {
        // Header integration styles
        style.textContent = `
          #page-translator {
            display: inline-flex;
            align-items: center;
            margin-left: 16px;
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 6px;
            padding: 4px 8px;
            font-family: inherit;
            font-size: 13px;
            backdrop-filter: blur(8px);
            position: relative;
          }

          .translator-toggle {
            background: none;
            border: none;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 4px;
            color: inherit;
          }

          .translator-toggle:hover {
            background: rgba(0,0,0,0.05);
          }

          .translator-dropdown {
            position: absolute;
            top: 100%;
            right: 0;
            margin-top: 4px;
            background: white;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            width: 200px;
            z-index: 1000;
            display: none;
          }

          .translator-dropdown.show {
            display: block;
          }
        `;
      } else {
        // Fallback styles (smaller floating widget)
        style.textContent = `
          #page-translator {
            position: fixed;
            top: 80px;
            right: 20px;
            z-index: 1000;
            background: white;
            border: 1px solid #e1e5e9;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            font-family: inherit;
            font-size: 14px;
            width: 200px;
          }
        `;
      }
      document.head.appendChild(style);
    }

    const translator = document.createElement('div');
    translator.id = 'page-translator';

    if (headerElement) {
      // Header integration version - compact dropdown
      translator.innerHTML = `
        <button class="translator-toggle" onclick="toggleTranslatorDropdown()">
          🌐 <span style="font-size: 12px;">▼</span>
        </button>
        <div class="translator-dropdown" id="translator-dropdown">
          <div style="font-weight: 500; margin-bottom: 8px; color: #333;">Translate Page</div>
          <select class="translator-select" id="target-language" style="width: 100%; padding: 6px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px;">
            <option value="">Select Language</option>
            <option value="Spanish">Spanish</option>
            <option value="French">French</option>
            <option value="German">German</option>
            <option value="Japanese">Japanese</option>
            <option value="Korean">Korean</option>
            <option value="Chinese">Chinese</option>
            <option value="Portuguese">Portuguese</option>
            <option value="Russian">Russian</option>
            <option value="Italian">Italian</option>
          </select>
          <button onclick="translatePage()" style="width: 100%; padding: 6px; background: #6FB000; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
            Translate with AI
          </button>
          <div style="font-size: 11px; color: #666; margin-top: 6px; line-height: 1.3;">
            Preserves code examples
          </div>
        </div>
      `;

      // Try to append to header element
      try {
        headerElement.appendChild(translator);
      } catch (e) {
        console.log('Could not append to header, using fallback');
        document.body.appendChild(translator);
      }
    } else {
      // Fallback version - simplified floating widget
      translator.innerHTML = `
        <div style="font-weight: 500; margin-bottom: 8px; color: #333; display: flex; align-items: center; gap: 6px;">
          🌐 <span>Translate</span>
        </div>
        <select class="translator-select" id="target-language" style="width: 100%; padding: 6px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px;">
          <option value="">Select Language</option>
          <option value="Spanish">Spanish</option>
          <option value="French">French</option>
          <option value="German">German</option>
          <option value="Japanese">Japanese</option>
          <option value="Korean">Korean</option>
          <option value="Chinese">Chinese</option>
        </select>
        <button onclick="translatePage()" style="width: 100%; padding: 6px; background: #6FB000; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 12px;">
          Translate
        </button>
      `;
      document.body.appendChild(translator);
    }
  }

  function extractPageForTranslation() {
    // Get main content area
    const main = document.querySelector('main, .content, article, .document') || document.body;

    // Extract headings
    const headings = Array.from(main.querySelectorAll('h1, h2, h3, h4, h5, h6'))
      .map(h => ({
        level: h.tagName,
        text: h.textContent.trim()
      }))
      .slice(0, 15); // First 15 headings

    // Extract main text paragraphs (exclude code blocks)
    const paragraphs = Array.from(main.querySelectorAll('p, li, td, .admonition-title'))
      .map(p => p.textContent.trim())
      .filter(text => text.length > 10 && text.length < 500) // Filter reasonable length
      .slice(0, 20); // First 20 paragraphs

    // Get page metadata
    const pageInfo = {
      title: document.title,
      url: window.location.href.split('?')[0], // Remove query params
      description: document.querySelector('meta[name="description"]')?.content || ''
    };

    return {
      pageInfo,
      headings,
      paragraphs,
      codeBlockCount: main.querySelectorAll('pre, .highlight').length
    };
  }

  function buildPageTranslationPrompt(content, targetLanguage) {
    return `Please translate this NeMo Run documentation page to ${targetLanguage}.

**CRITICAL TRANSLATION RULES:**
- Translate ONLY explanatory text, headings, and descriptions
- Keep ALL code examples completely unchanged (preserve syntax, variable names, function names)
- Preserve technical terms that are standard in programming (e.g., "API", "JSON", "Docker")
- Maintain all formatting, links, and structure
- Keep NeMo Run terminology consistent (translate "guide" but keep "NeMo Run")

**Page Information:**
Title: ${content.pageInfo.title}
URL: ${content.pageInfo.url}
${content.pageInfo.description ? `Description: ${content.pageInfo.description}` : ''}

**Page Structure (${content.headings.length} sections):**
${content.headings.map(h => `${h.level}: ${h.text}`).join('\n')}

**Main Content Excerpts:**
${content.paragraphs.slice(0, 10).map((p, i) => `${i+1}. ${p}`).join('\n\n')}

**Additional Info:**
- This page contains ${content.codeBlockCount} code examples that should remain untranslated
- Focus on making the explanatory content accessible to ${targetLanguage} speakers
- Maintain professional technical documentation tone

Please provide a comprehensive translation that helps ${targetLanguage} speakers understand NeMo Run concepts while keeping all technical elements intact.`;
  }

  // Make translatePage available globally
  window.translatePage = function() {
    const targetLanguage = document.getElementById('target-language').value;
    if (!targetLanguage) {
      alert('Please select a target language first.');
      return;
    }

    const content = extractPageForTranslation();
    const prompt = buildPageTranslationPrompt(content, targetLanguage);

    copyToClipboard(prompt).then(() => {
      alert(`🌐 Translation prompt copied!\n\nThis will translate the page content to ${targetLanguage} while preserving all code examples.\n\nChatGPT will open next - paste the prompt to get your translation.`);
      openChat();
    }).catch(() => {
      alert('Failed to copy to clipboard. Please try again.');
    });
  };

  // Make toggle function available globally
  window.toggleTranslatorDropdown = function() {
    const dropdown = document.getElementById('translator-dropdown');
    if (dropdown) {
      dropdown.classList.toggle('show');
    }
  };

  // Close dropdown when clicking outside
  document.addEventListener('click', function(event) {
    const translator = document.getElementById('page-translator');
    const dropdown = document.getElementById('translator-dropdown');
    if (dropdown && translator && !translator.contains(event.target)) {
      dropdown.classList.remove('show');
    }
  });

  ready(function () {
    createPageTranslator();
  });
})();
