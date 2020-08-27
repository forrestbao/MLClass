# Beamer to Markdown (that Pandoc can later convert into PDF via LateX/Beamer)

To convert from Beamer/LaTeX to Markdown, just do `pandoc -s slides.tex -o slides.md`. This will give you a Markdown file to begin with (It cannot be used to produce PDF slides). If you run into problem with the `<>` directive of Beamer, e.g., ` \section<presentation>*{Outline}`, just remove that block. 

Then some manual final-touches:

1. Pandoc does not seem to always treat a slide title in Beamer as a section name, e.g., when the LaTeX source has `\section`. So the resulting lines for slide title becomes something like `[This is my title] {}` in Markdown. Please manually move all slide title to the top level, e.g., `# This is a slide title`. 

2. Pandoc moves all LaTeX footnotes to the end of the Markdown output. Please maually move them back to corresponding pages. 

3. Pandoc does not seem to convert columns in Beamer well. Please [manually fix](https://pandoc.org/MANUAL.html#columns).

4. Please organize all pictures used in one set of slides into a folder called `figs` and please the folder in the same level/directory with the Markdown souce. 

5. Check whether any slide overflows, e.g., some content in the source is not in the PDF. To see whether the slides work, `pandoc -t beamer -s slides.md -o slides.pdf`.  If overflows, just manually break by creating a new slide with the title `slide title (cont.)`. 
