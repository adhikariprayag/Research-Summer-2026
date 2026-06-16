# spaCy Sample NER Observations

## R001_standard

For `R001_standard`, spaCy extracted many entities from the clean standard resume, but not all labels were correct. It correctly detected locations such as Bloomfield, NJ, and Kathmandu, and it also detected several real organizations such as Google Developers Group, Caldwell University, Office of Online Education, Google, Coursera, Broadway Infosys, and NASA Small-Body Database. However, many technical skills and tools such as Canva, Figma, CSS, HTML, JavaScript, GitHub, React.js, and Bootstrap were labeled as PERSON or ORG instead of being treated as SKILL. This shows that even with a clean standard layout, spaCy’s default NER model is not enough for resume skill extraction. Overall, R001 had the cleanest extraction, but it still needs a custom SKILL matcher.

## R004_creative_text

For `R004_creative_text`, spaCy correctly detected the applicant name Taro Tanaka and locations such as Tokyo, Japan, and Kyoto, Japan. It also detected some organizations such as AAA Company, BBB University, and IELTS. However, the creative two-column layout and contact icons introduced some noise, such as the artifact `Ȳ` being labeled as ORG. Several technical skills and tools, including Python, BigQuery, AWS Deequ, GCP, Oracle, Docker, GitLab, and AWS, were labeled as ORG instead of SKILL. Some labels were clearly wrong, such as Flask and Docker being labeled as PERSON, and Scala being labeled as GPE. Date extraction was also incomplete because some date ranges were captured partially, such as `2019 - Mar 2020`, instead of the full job date context. Overall, R004 shows that the creative layout is machine-readable, but artifacts and layout structure create more NER errors than the standard resume.

## R005_complex_text

For `R005_complex_text`, spaCy correctly detected Jack Sparrow as a PERSON several times and detected some real entities such as Tortuga Uni, London, Tortuga Printing Press, The Pragmatic Pirate, and dates like 1729 and Nov. 1726. However, this file had the noisiest NER output. Many placeholder words such as `Lorem` were repeatedly labeled as ORG, and visual symbols such as `○` were labeled as GPE. Some broken or incorrect text, such as `sparrow`, `consectetur adipiscing elit`, `Certificates`, and `Lady Swan`, were labeled as PERSON. This shows that the complex visual resume created many artifacts and confusing text fragments. Overall, R005 is the strongest example of how complex formatting can produce noisy extracted text and reduce the reliability of spaCy NER.

## SKILL Matcher Note

A custom skill matcher was added because spaCy’s default NER model mislabeled many technical skills as PERSON or ORG. The matcher searches the extracted resume text for a predefined list of CS/design skills and saves them as SKILL entities.

## SKILL Matcher Inspection Note

The SKILL keyword matcher successfully extracted several technical skills from R004 and R005, but the output still needs manual inspection. Some extracted skills are accurate even when they appear outside the formal Skills section because they are still technical skills mentioned in the resume text. However, some expected skills were missed because they were not included in the predefined skill list, had unusual formatting, or appeared in broken extracted text. This shows that the keyword matcher improves skill detection compared to spaCy alone, but it is still limited by the quality of the skill list and the extracted PDF text.

## SKILL Matcher Manual Inspection Note 

The SKILL keyword matcher improved resume skill detection by identifying technical and design tools that spaCy’s default NER model often mislabeled as PERSON or ORG. For R001, the matcher produced the strongest output and detected many valid web development, design, and publishing tools. For R004, the matcher correctly detected several technical skills such as Python, JavaScript, HTML, CSS, Docker, AWS, GCP, Linux, BigQuery, Oracle, Flask, and Scala, but it missed expected skills such as C++ and GitLab. For R005, the matcher detected Python, JavaScript, HTML, CSS, and Linux, but missed some expected skills such as R, LATEX, and Android. Overall, the matcher works better than spaCy alone for SKILL extraction, but it still depends heavily on the completeness of the predefined skill list and the quality of the extracted PDF text.

## The R004 C++ Issue
For R004, the matcher successfully detected most expected technical skills, including GitLab through the more specific term GitLab CI. However, C++ was initially missed because symbols like “++” do not work well with the simple word-boundary regex pattern. This shows that some programming languages and tools require special regex handling, especially when the skill name contains symbols such as +, #, or . .

## Regex Update Note

The SKILL matcher was updated after C++ was not detected in R004. The original regex used simple word boundaries, which worked for normal skill names like Python, Docker, and Oracle, but failed for skills containing symbols such as C++. The regex was changed to handle special characters more reliably. After this update, C++ was successfully extracted from R004. This shows that skill extraction requires careful handling of programming languages and tools that contain symbols, such as C++, C#, .NET, and Node.js.

## Combined Entity Output Note

The spaCy NER output and the custom SKILL matcher output were successfully combined into one final entity CSV per resume. The spaCy output provides PERSON, ORG, DATE, and GPE entities, while the custom matcher adds SKILL entities. Duplicate rows were removed during the merge, so the final combined counts are lower than the raw spaCy and skill counts. This creates a cleaner entity list for each resume and prepares the project for manual evaluation against the Gold Standard annotations.