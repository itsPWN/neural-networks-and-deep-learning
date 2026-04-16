# Offline Book: Neural Networks and Deep Learning

This directory contains an offline copy of Michael Nielsen's free online book
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/).

## How to Read

Open any chapter in your web browser:

| File | Content |
|------|---------|
| [index.html](index.html) | Table of contents |
| [about.html](about.html) | What this book is about |
| [exercises_and_problems.html](exercises_and_problems.html) | On the exercises and problems |
| [chap1.html](chap1.html) | Ch 1: Using neural nets to recognize handwritten digits |
| [chap2.html](chap2.html) | Ch 2: How the backpropagation algorithm works |
| [chap3.html](chap3.html) | Ch 3: Improving the way neural networks learn |
| [chap4.html](chap4.html) | Ch 4: A visual proof that neural nets can compute any function |
| [chap5.html](chap5.html) | Ch 5: Why are deep neural networks hard to train? |
| [chap6.html](chap6.html) | Ch 6: Deep learning |
| [sai.html](sai.html) | Appendix: Is there a simple algorithm for intelligence? |
| [acknowledgements.html](acknowledgements.html) | Acknowledgements |
| [faq.html](faq.html) | Frequently Asked Questions |

**Quick open from terminal** (from repo root):

    open book/chap1.html        # macOS
    xdg-open book/chap1.html    # Linux
    start book\chap1.html       # Windows

Or if you're already inside the `book/` directory:

    open chap1.html             # macOS

## Fully Offline

Everything works without an internet connection:

- **Math equations** -- MathJax 2.7.1 is bundled locally in `mathjax/`
- **Interactive demos** -- Chapter 3 sliders, Chapter 4 neuron visualizations
  and videos, Chapter 5 gradient plots -- all JS/CSS/video files included
- **Diagrams and figures** -- all images in `images/`

The only things that won't render offline are two small PayPal donate
button images in the sidebar (the author's donation links).

## License

The book content is (c) Michael A. Nielsen, 2015, published by Determination
Press, and licensed under [Creative Commons Attribution-NonCommercial 3.0
Unported](https://creativecommons.org/licenses/by-nc/3.0/). This copy is
included for educational use alongside the code exercises.
