// Preprocess mermaid blocks, then dynamically load and render mermaid.
// Loading mermaid dynamically (instead of via extra_javascript) prevents
// its auto-init from racing with our DOM preprocessing.
document.addEventListener("DOMContentLoaded", function () {
  // fence_code_format produces <pre class="mermaid"><code>...</code></pre>.
  // Extract the decoded text from <code>, remove the <code> wrapper, and
  // set it directly on the <pre> so mermaid can render SVG properly.
  var blocks = document.querySelectorAll("pre.mermaid");
  if (blocks.length === 0) return;

  blocks.forEach(function (pre) {
    var code = pre.querySelector("code");
    if (!code) return;
    var source = code.textContent;
    // Fix dotted Python module paths (e.g. "pkg.mod.Class") which are
    // invalid mermaid node IDs from mkdocstrings inheritance diagrams.
    source = source.replace(
      /\b([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+)(?=[\[\(\{<\s]|$)/gm,
      function (m) { return m.replace(/\./g, "__"); }
    );
    pre.removeChild(code);
    pre.textContent = source;
  });

  // Dynamically load mermaid and render
  var script = document.createElement("script");
  script.src = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js";
  script.onload = function () {
    mermaid.initialize({ startOnLoad: false, theme: "default" });
    mermaid.run({ querySelector: "pre.mermaid" });
  };
  document.head.appendChild(script);
});
