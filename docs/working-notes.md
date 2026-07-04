# Working Notes — Author Instructions

Every team that submitted a system to GenSIE 2026 must also submit a **working notes paper** describing it. This page is the complete set of rules for that paper: format, length, citations, copyright, deadlines, and how to send it to us.

These rules follow the IberLEF 2026 author guidelines (issued by the IberLEF general chairs, 2026-04-22). If anything below conflicts with the IberLEF guidelines, the IberLEF guidelines win — please email us and we will update this page.

---

## 1. Format and length

- **Style:** CEURART, 1-column LaTeX. Template: <https://ceur-ws.org/Vol-XXX/CEURART.zip>. Always download a fresh copy — CEUR updates the template frequently. Word and ODT variants exist but are **strongly discouraged** by CEUR itself; please use LaTeX.
- **Length:** **5 pages minimum**, no maximum.
- **Language:** English only.
- **Copyright year:** `\copyrightyear{2026}` in the preamble.
- **Conference command:** `\conference{IberLEF 2026, September 2026, Le\'on, Spain}`.
- **No page numbers, no headers.** The first page may carry only the mandatory copyright footnote. Other-page footnotes are allowed but please keep them rare.
- **Authors:** full names, no initials (e.g. `María García`, not `M. García`). Each author's affiliation must include university and country.
- **Title:** emphatic capital English notation — `Filling an Author Agreement by Autocompletion`, *not* `Filling an author agreement by autocompletion`.

## 2. Mandatory "Declaration on Generative AI" section

CEURART requires a **Declaration on Generative AI** section in every paper. Fill it honestly with the use you actually made of generative AI when preparing the paper or the system. Leaving it out will get the paper bounced at the IberLEF level — out of our hands.

## 3. Citations you must include

Every working note must cite **two** overviews:

1. **The GenSIE 2026 task overview** (our paper). The final bibtex will be published on this page once our overview is camera-ready. Until then, please use the placeholder bibkey `gensie2026overview` and we will swap in the final citation during the review round. A draft entry is:

    ```bibtex
    @inproceedings{gensie2026overview,
      title     = {Overview of GenSIE at IberLEF 2026: Schema-Guided Information Extraction with Small Language Models in Spanish},
      author    = {Piad-Morffis, Alejandro and others},
      booktitle = {Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026), co-located with the 42nd Conference of the Spanish Society for Natural Language Processing (SEPLN 2026), CEUR-WS.org},
      year      = {2026}
    }
    ```

2. **The IberLEF 2026 general overview** (Bonet-Jover, González-Barba, Chiruzzo):

    APA:
    > Alba Bonet-Jover, González-Barba, J. A., Chiruzzo, L. (2026). In Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026), co-located with the 42nd Conference of the Spanish Society for Natural Language Processing (SEPLN 2026), CEUR-WS.org

    Bibtex:
    ```bibtex
    @inproceedings{iberlef2026overview,
      title     = {Overview of IberLEF 2026: Natural Language Processing Challenges for Spanish and other Iberian Languages},
      author    = {Bonet-Jover, Alba and Gonz{\'a}lez-Barba, Jos{\'e} {\'A}ngel and Chiruzzo, Luis},
      booktitle = {Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2026), co-located with the 42nd Conference of the Spanish Society for Natural Language Processing (SEPLN 2026), CEUR-WS.org},
      year      = {2026}
    }
    ```

## 4. CEUR copyright agreement — physical signature required

CEUR-WS requires **at least one author of each paper to sign the CEUR copyright agreement on paper, with a blue pen, and scan it.** Digital signatures are not accepted. There are two variants:

- **NTP** (no third-party copyrighted material): use this in the vast majority of cases.
- **TP** (your paper or accompanying materials reuse third-party copyrighted material): use this if you reproduced figures, tables, code, or data from another work — and **also attach the rightsholder's written permission**.

Fill the form with:

- **Name and year of the event:** `IberLEF 2026`
- **Editors of the proceedings:** Alba Bonet-Jover, José Ángel Gónzalez-Barba, Luis Chiruzzo, Alejandro Piad Morffis, Hugo Jair Escalante, Iker de la Iglesia, Albina Sarymsakova, Fazlourrahman Balouchzahi, Alba María Mármol Romero, Horacio Saggion, Tania Gisela Alcántara Medina, Ansel Yoan Rodríguez González, María Victoria Cantero Romero, Tomás Bernal-Beltrán, Álvaro Rodrigo, Miguel Ángel Álvarez Carmona, Vicent Ahuir, Luis Israel Ramos Pérez, Niels Martínez Guevara

Print → sign → scan → bundle the scan with your paper PDF (see §6).

> **Plan around the post.** International shipping is not in the loop here — only printing and scanning — but if you do not have a printer/scanner readily available, line that up *now*, not on the last day.

## 5. Deadlines

GenSIE participants are on a **tighter internal schedule than IberLEF**, because we owe IberLEF the full bundle (overview + every working note + every signed agreement) on **2026-07-03**, and we want time to review each paper before it goes out under our task's name.

| Deadline                          | Date         | What happens                                                                 |
| --------------------------------- | ------------ | ---------------------------------------------------------------------------- |
| **Recommended submission**        | **2026-06-10** | We do one full review round and return comments within ~5 days. You revise.  |
| **Absolute submission (hard)**    | **2026-06-19** | Final, no review. If we reject the paper at this point, **the team loses its slot in IberLEF 2026.** |
| Final revisions back to us        | 2026-06-26   | For papers that went through the June 10 review.                             |
| IberLEF bundle uploaded by us     | 2026-07-03   | The IberLEF general chairs' hard deadline.                                   |

We strongly recommend hitting the **June 10** date. It is the only path that gives you a review round; once you cross into the June 19 window, the paper either lands clean or doesn't land at all.

**What counts as "rejection".** A working note can be rejected for: not meeting the format rules in §1; missing or insincere Declaration on Generative AI; a misrepresentation of results that the system did not actually produce; or material that doesn't describe a system that participated in GenSIE 2026. Stylistic feedback is *not* a rejection — we will simply suggest fixes during the June 10 review round.

## 6. How to submit

Open a comment on your team's existing `[SUBMISSION]` issue in <https://github.com/gia-uh/gensie> with:

1. A link to a PDF of your working note (any of: a release attachment on your private submission repo, a direct file URL, or a private gist).
2. A link to the scanned PDF of the signed CEUR agreement (same channel).
3. The bibtex entry for any new citation you added that is not already in your paper's references.

> If a link-based handoff doesn't work for you (e.g. your institution blocks file shares), email both PDFs to `apiad@matcom.uh.cu` with subject `GenSIE 2026 — Working Notes — <Team Name>` and we will pick them up from there.

We will not chase missing submissions. If the working notes for your team are not in our hands by **2026-06-19 at 23:59 anywhere-on-earth**, your team will not appear in the IberLEF 2026 proceedings.

## 7. Questions

Reply on your `[SUBMISSION]` issue, or email Alejandro Piad Morffis (`apiad@matcom.uh.cu`). English or Spanish, whatever you prefer.

— The GenSIE 2026 Organizing Committee
(GIA-UH, University of Havana · GPLSI, University of Alicante)
