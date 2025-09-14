const translations = {
  en: "<strong>Made with love for Python</strong><br>Writing to remember. Sharing to help.",
  ru: "<strong>Создано с любовью к Python</strong><br>Пишу, чтобы запомнить. Делюсь, чтобы помочь.",
};

const lang = document.documentElement.lang || "en";
const TARGET_FOOTER_TEXT = translations[lang] || translations["en"];

function updateFooter() {
  const copyrightDiv = document.querySelector(".md-copyright");
  if (!copyrightDiv) return;

  if (copyrightDiv.innerHTML.trim() === TARGET_FOOTER_TEXT.trim()) return;

  copyrightDiv.innerHTML = TARGET_FOOTER_TEXT;
}

document.addEventListener("DOMContentLoaded", updateFooter);

document.addEventListener("click", function (e) {
  if (e.target.closest("a")) {
    setTimeout(updateFooter, 100);
  }
});
