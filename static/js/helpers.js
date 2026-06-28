// MAIZE-XNet — optional client-side helpers
// Auto-scroll to results after analysis
function scrollToResults() {
  const el = document.querySelector('.result-hero');
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Add terminal cursor blink to header
document.addEventListener('DOMContentLoaded', function () {
  const logo = document.querySelector('.logo-wordmark');
  if (logo) {
    const cursor = document.createElement('span');
    cursor.style.cssText = 'color:#39ff88;animation:blink 1s step-end infinite;margin-left:2px;';
    cursor.textContent = '_';
    logo.appendChild(cursor);
    const style = document.createElement('style');
    style.textContent = '@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}';
    document.head.appendChild(style);
  }
});
