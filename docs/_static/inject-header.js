// Inject shared XFEL documentation header
(function() {
  // For local testing uncomment the line below to use relative paths and
  // comment out the Github URL.
  // const baseUrl = '';
  const baseUrl = 'https://raw.githubusercontent.com/European-XFEL/EXtra/refs/heads/master/docs';
  const headerUrl = `${baseUrl}/xfel-header.html`;
  const cssUrl = `${baseUrl}/css/xfel-header.css`;

  let htmlContent = null;
  let cssReady = false;

  // Inject CSS once and make sure it stays
  async function ensureCSS() {
    if (cssReady && document.getElementById('xfel-header-style')) {
      return;
    }

    if (document.getElementById('xfel-header-style')) {
      cssReady = true;
      return;
    }

    try {
      const response = await fetch(cssUrl);
      const css = await response.text();
      const style = document.createElement('style');
      style.id = 'xfel-header-style';
      style.textContent = css;
      document.head.appendChild(style);
      cssReady = true;
    } catch (error) {
      console.warn('Error loading XFEL header CSS:', error);
      // Continue anyway to not block header injection
    }
  }

  // Inject the header HTML
  async function injectHeader() {
    // For MkDocs Material theme - create placeholder to prevent layout shift
    const mdHeader = document.querySelector('.md-header');
    let placeholder = document.getElementById('xfel-header-placeholder');

    if (mdHeader && !placeholder) {
      // Create a placeholder div to reserve space and match header background
      // Inline styles prevent layout shift before CSS loads
      placeholder = document.createElement('div');
      placeholder.id = 'xfel-header-placeholder';
      // Note: the height and background color should match `xfel-header-height`
      // and `xfel-header-bg-color` in `xfel-header.css`.
      placeholder.style.cssText = 'height: 40px; min-height: 40px; background-color: #1a237e; width: 100%;';
      mdHeader.parentNode.insertBefore(placeholder, mdHeader);
    }

    // Remove existing header if present
    const existing = document.querySelector('.xfel-cross-nav');
    if (existing) {
      existing.remove();
    }

    // Ensure CSS is loaded before inserting HTML
    await ensureCSS();

    // If we have cached HTML, use it
    if (htmlContent) {
      insertHeaderHTML(htmlContent);
      return;
    }

    // Otherwise fetch it
    try {
      const response = await fetch(headerUrl);
      if (!response.ok) {
        console.warn('Failed to load XFEL header:', response.status);
        return;
      }
      const html = await response.text();
      htmlContent = html;
      insertHeaderHTML(html);
    } catch (error) {
      console.warn('Error loading XFEL header:', error);
    }
  }

  function insertHeaderHTML(html) {
    const nav = document.createElement('div');
    nav.className = 'xfel-cross-nav';
    nav.innerHTML = html;

    // For MkDocs Material theme - replace placeholder or inject before header
    const header = document.querySelector('.md-header');
    const placeholder = document.getElementById('xfel-header-placeholder');

    if (header) {
      if (placeholder) {
        // Replace the placeholder to avoid layout shift
        placeholder.replaceWith(nav);
      } else {
        header.parentNode.insertBefore(nav, header);
      }
    } else {
      // Fallback: inject at the top of the body (for Sphinx and others)
      document.body.insertBefore(nav, document.body.firstChild);
    }
  }

  // Initial load
  document.addEventListener('DOMContentLoaded', function() {
    injectHeader();

    // Handle instant navigation in MkDocs Material
    // The library exposes a document$ observable when instant navigation is enabled
    if (typeof document$ !== 'undefined') {
      let firstEmission = true;
      document$.subscribe(function() {
        // Skip the first emission as we already injected on DOMContentLoaded
        if (firstEmission) {
          firstEmission = false;
          return;
        }
        injectHeader();
      });
    }
  });
})();
