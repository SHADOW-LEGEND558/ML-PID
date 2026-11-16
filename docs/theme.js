/* Apply a saved preference before paint; storage can be disabled in private mode. */
(() => {
  let theme = 'light';
  try { if (localStorage.getItem('ml-pid-theme') === 'dark') theme = 'dark'; } catch (_) { /* Keep the default. */ }
  const apply = value => {
    theme = value;
    document.documentElement.dataset.theme = theme;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', theme === 'dark' ? '#171717' : '#fafaf9');
    const button = document.getElementById('theme-toggle');
    if (button) {
      button.setAttribute('aria-pressed', String(theme === 'dark'));
      button.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
      document.getElementById('theme-label').textContent = `${theme === 'dark' ? 'Light' : 'Dark'} mode`;
    }
  };
  apply(theme);
  document.addEventListener('DOMContentLoaded', () => {
    apply(theme);
    document.getElementById('theme-toggle').addEventListener('click', () => {
      apply(theme === 'dark' ? 'light' : 'dark');
      try { localStorage.setItem('ml-pid-theme', theme); } catch (_) { /* Theme still works for this page. */ }
    });
  });
})();
