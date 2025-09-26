// WCAG-compliant interaction for architecture mermaid diagram
// Extracted from docs/about/architecture.md into a reusable extension asset

document.addEventListener('DOMContentLoaded', function () {
  const clickableIds = ['architecture-diagram', 'architecture-mermaid'];

  clickableIds.forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;

    // Make the container operable via keyboard and announced clearly
    el.setAttribute('tabindex', '0');
    el.setAttribute('role', 'button');
    el.setAttribute(
      'aria-label',
      'Open architecture diagram: NeMo Run core architecture showing configuration, execution, and management layers'
    );

    el.addEventListener('click', function () {
      // Create modal
      const modal = document.createElement('div');
      modal.className = 'diagram-modal';
      modal.setAttribute('role', 'dialog');
      modal.setAttribute('aria-modal', 'true');
      modal.setAttribute('aria-labelledby', 'architecture-modal-title');
      modal.innerHTML = `
        <div class="diagram-modal-content">
          <button class="diagram-modal-close" aria-label="Close dialog" type="button">&times;</button>
          <h3 id="architecture-modal-title">NeMo Run Core Architecture</h3>
          <div style="max-height: 80vh; overflow-y: auto;">${this.innerHTML}</div>
        </div>
      `;

      document.body.appendChild(modal);
      modal.style.display = 'block';

      // Focus management: trap focus within modal and restore on close
      const previousFocus = document.activeElement;
      const getFocusable = (root) =>
        Array.from(
          root.querySelectorAll(
            'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
          )
        );

      const closeBtn = modal.querySelector('.diagram-modal-close');
      closeBtn && closeBtn.focus();

      const trapHandler = (e) => {
        if (e.key !== 'Tab') return;
        const focusables = getFocusable(modal);
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      };
      modal.addEventListener('keydown', trapHandler);

      // Remove unwanted buttons from within the modal (Copy / Explain / Ask AI)
      (function removeUnwantedButtons(root) {
        root.querySelectorAll('.ai-toolbar').forEach((el) => el.remove());
        root.querySelectorAll('.copybtn').forEach((el) => el.remove());

        const elements = root.querySelectorAll('button, a');
        elements.forEach((el) => {
          const label = `${el.getAttribute('aria-label') || ''} ${el.getAttribute('title') || ''} ${
            el.textContent || ''
          }`.trim();
          if (/\b(copy|explain|ask ai)\b/i.test(label) || el.classList.contains('copybtn')) {
            el.style.display = 'none';
          }
        });
      })(modal);

      // Also hide/remove toolbars in the inline diagram areas
      (function removeFromInlineDiagrams() {
        clickableIds.forEach((cid) => {
          const container = document.getElementById(cid);
          if (!container) return;
          container.querySelectorAll('.ai-toolbar, .copybtn').forEach((el) => el.remove());
          container.querySelectorAll('.ai-btn').forEach((el) => (el.style.display = 'none'));
        });
      })();

      // Close modal functionality
      const doClose = () => {
        modal.removeEventListener('keydown', trapHandler);
        modal.remove();
        if (previousFocus && previousFocus.focus) previousFocus.focus();
      };
      closeBtn && closeBtn.addEventListener('click', doClose);

      // Close on outside click
      modal.addEventListener('click', function (e) {
        if (e.target === modal) {
          doClose();
        }
      });

      // Close on Escape key
      document.addEventListener('keydown', function onEsc(e) {
        if (e.key === 'Escape') {
          doClose();
          document.removeEventListener('keydown', onEsc);
        }
      });
    });

    // Keyboard activation for trigger
    el.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        el.click();
      }
    });
  });

  // Mermaid-specific enhancements: add accessible roles/labels to rendered SVGs
  const mermaidContainer = document.getElementById('architecture-mermaid');
  if (mermaidContainer) {
    const svg = mermaidContainer.querySelector('svg');
    if (svg) {
      // Mark the diagram as an image with a descriptive label
      svg.setAttribute('role', 'img');
      if (!svg.getAttribute('aria-label')) {
        svg.setAttribute(
          'aria-label',
          'NeMo Run core architecture diagram showing configuration, execution, and management layers with flows between them'
        );
      }
      // Ensure background remains transparent for contrast
      svg.style.backgroundColor = 'transparent';
    }
  }
});
