# Sample Evaluation Observations

The Gold Standard comparison was completed successfully. The script compared the manually created Gold Standard answer key against the combined outputs from spaCy and the custom SKILL matcher.

The initial exact-match scores are low because the current evaluation only counts an entity as correct when both the entity text and entity type match exactly. This means partial matches and label mismatches are counted as errors.

For R001_standard, several entities were correctly matched, including Google Developers Group, Caldwell University, Office of Online Education, NASA Small-Body Database, Google, Broadway Infosys, Oct 2025, and April 2026. However, many skills such as Canva, Figma, HTML, JavaScript, and GitHub were also extracted by spaCy with incorrect labels such as ORG or PERSON. This confirms that spaCy alone is not reliable for SKILL extraction.

For R004_creative_text, the system correctly matched Taro Tanaka, AAA Company, BBB University, IELTS, and 4 years. However, several locations and dates were only partially extracted. For example, Tokyo, Japan was split into separate entities, and date ranges were not always captured completely. Some skills were also mislabeled by spaCy, such as Python as ORG, Scala as GPE, and Flask as PERSON.

For R005_complex_text, the system correctly matched Jack Sparrow, London, Tortuga Uni, Tortuga Printing Press, The Pragmatic Pirate, 2019, 1729, and Nov. 1726. However, the complex visual layout created noisy false entities such as Lorem, sparrow, B.A., and symbol artifacts. This shows that complex formatting can create both missed entities and incorrect extractions.

Overall, these results should be treated as early exact-match pilot evaluation results. The next step is to identify partial matches and boundary errors separately instead of treating them the same as completely wrong outputs.