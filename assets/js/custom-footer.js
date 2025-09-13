function updateFooter() {
  const copyrightDiv = document.querySelector(".md-copyright");
  if (copyrightDiv) {
    copyrightDiv.innerHTML = `
            <strong>Создано с любовью к Python</strong><br>
            Пишу, чтобы запомнить. Делюсь, чтобы помочь.
        `;
  }
}

document.addEventListener("DOMContentLoaded", updateFooter);
document.addEventListener("mkdocs:page-loaded", updateFooter);
