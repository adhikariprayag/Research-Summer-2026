# Text-Based PDF Check Notes

## Purpose

After advisor feedback, I collected new creative and complex resume PDFs that were not converted from PNG. The purpose of this check was to confirm whether these new PDFs contain machine-readable text before using them in the main experiment.

## Result

The new resume PDFs, R004 and R005, produced more than 0 text characters and more than 0 words when tested with PyMuPDF. This confirms that they are machine-readable PDFs and are usable for the main layout-based extraction test.

## Interpretation

This is important because my earlier creative and complex resumes were converted from PNG files and appeared to PyMuPDF as image-based PDFs. Those files were useful as a pilot failure case, but they are not ideal for the main experiment because they test image-based extraction failure rather than layout-based NER degradation.

The new R004 and R005 files are better for the main experiment because they allow me to test whether creative layouts affect extraction order, text fragmentation, and later NER performance while still preserving an actual text layer.

## Research Decision

For the main experiment, I will use text-based PDFs whenever possible. PNG-converted or image-based PDFs will be documented separately as an early pilot observation, but they will not be treated as the main creative resume condition.