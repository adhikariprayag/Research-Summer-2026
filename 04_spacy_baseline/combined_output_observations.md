# Combined Entity Output Observations

The combined entity outputs successfully include PERSON, ORG, DATE, GPE, and SKILL entities for all three sample resumes. This is an improvement over spaCy alone because the custom matcher adds technical and design skills that spaCy often mislabeled as PERSON or ORG.

For R001_standard, the combined file contains the largest number of extracted entities. This is expected because the resume has a clean standard layout and many clearly listed tools, organizations, projects, dates, and locations. The combined output is useful, but it still contains some questionable spaCy labels that will need manual checking.

For R004_creative_text, the combined file includes PERSON, ORG, DATE, GPE, and SKILL entities from the creative two-column resume. The skill matcher improved detection of technical tools such as Python, C++, Docker, AWS, GCP, BigQuery, Oracle, GitLab CI, Flask, and Scala. However, the resume’s layout and contact icon artifacts still caused some noisy or questionable spaCy entities. This supports the observation that creative resumes can be machine-readable but still produce NER issues.

For R005_complex_text, the combined file includes all target entity categories, but the output remains the noisiest. The skill matcher correctly helped recover programming skills such as HTML, CSS, Python, R, JavaScript, LaTeX, Android, and Linux. However, spaCy still produced many questionable entities because the complex visual resume contains placeholder text, symbols, repeated sections, and font artifacts. This makes R005 the strongest early example of layout complexity reducing NER reliability.

Overall, the combined output step worked successfully. The project now has one entity file per resume containing spaCy entities plus custom SKILL entities. The next step is to manually compare these combined outputs against the actual resume content and mark entities as correct, missed, or incorrect.