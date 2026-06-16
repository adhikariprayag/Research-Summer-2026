# Week 4 Evaluation Observations

The Week 4 exact-match evaluation produced an overall precision of 0.303, recall of 0.374, and F1-score of 0.335.

Manual inspection of R006_kaggle showed that most false-positive outputs were ORG entities. spaCy incorrectly classified many technical skills and software terms, including Android, Azure, Bash, Cisco ASA, CSS, DHCP, DNS, HTML, Java, Linux, PowerShell, SQL, VMware, and VPN, as organizations.

R006 also contained 49 missed SKILL entities. Several were not included in the current keyword matcher, including BaaS, Cacti, Data Domain, DFS, Exchange Online, Group Policy Objects, iSCSI, LDAP, Nmap, STaaS, TwinStrata, Windows Server versions, and Zenmap.

Nineteen entities were detected by spaCy but assigned the wrong entity type. For example, Active Directory Federation Services, ADFS, Apple iOS, Bash, Cisco ASA, TFS, VDI, and Wi-Fi were marked as ORG instead of SKILL. Under strict evaluation, these create both a false-positive ORG and a missed SKILL.

These findings show that the low baseline score is influenced by both general-purpose spaCy label errors and incomplete coverage in the custom SKILL lexicon. The Gold Standard should remain unchanged. The next methodological improvement should expand the SKILL dictionary and separately report per-entity-type performance and label-mismatch errors.

## OCR Resume Error Analysis — R002 and R003
R002_creative and R003_complex were image-based resumes that originally produced no readable text with PyMuPDF. OCR using Tesseract recovered 1,945 characters from R002 and 2,763 characters from R003, allowing both resumes to be processed by spaCy and the custom skill matcher.

For R002_creative, the combined system produced 25 entities. Compared with 38 Gold Standard entities, 6 were correct, 32 were missed, and 19 were incorrect. Precision was 0.240, recall was 0.158, and F1 was 0.190. The six correct entities were technical skills such as C#, CSS, HTML, Java, Python, and React. OCR introduced errors in dates, including “Sep 2078” instead of “Sep 2018” and “Nov 2027” instead of “Nov 2021.” spaCy also incorrectly labeled headings and text fragments as organizations or people. Many design and software skills were missed because they were not included in the current skill dictionary.

For R003_complex, the combined system produced 19 entities. Compared with 36 Gold Standard entities, 8 were correct, 28 were missed, and 11 were incorrect. Precision was 0.421, recall was 0.222, and F1 was 0.291. Correct entities included Melissa Ware, Burton Agency, Facebook, Instagram, The Ohio State University, and several durations. However, all writing and marketing skills, such as Copywriting, SEO, Digital Marketing, Editing, and Creative Writing, were missed because the current skill dictionary mainly contains technical skills. Location fragments such as “Columbus” and “Ohio” were also extracted separately instead of matching the Gold Standard entity “Columbus, Ohio.”

Overall, the two OCR resumes achieved a precision of 0.318, recall of 0.189, and F1 of 0.237. OCR made previously unreadable resumes processable, but the results show that errors came from three stages: OCR character and layout errors, spaCy entity-labeling errors, and an incomplete skill dictionary. Therefore, resume parsing failure in creative image-based resumes is a pipeline-level problem rather than only an OCR problem.

#### Final 20-Resume Baseline Evaluation

The final evaluation included 20 resumes: 10 standard and 10 creative or complex resumes. The Gold Standard contained 1,108 manually annotated entities across the PERSON, ORG, DATE, GPE, and SKILL categories. The combined spaCy and custom skill-matcher system generated 1,277 unique entities.

For the 10 standard resumes, the system produced 147 correct entities, missed 475 Gold Standard entities, and generated 551 incorrect entities. This resulted in a precision of 0.211, recall of 0.236, and F1 score of 0.223.

For the 10 creative resumes, the system produced 175 correct entities, missed 311 Gold Standard entities, and generated 404 incorrect entities. This resulted in a precision of 0.302, recall of 0.360, and F1 score of 0.329.

Across all 20 resumes, the system produced 322 correct entities, missed 786 entities, and generated 955 incorrect entities. Overall precision was 0.252, recall was 0.291, and F1 was 0.270.

Contrary to the initial expectation, the creative-resume group achieved higher scores than the standard-resume group. This result should not yet be interpreted as evidence that creative resumes are easier to parse. The result may be affected by resume length, entity density, differences in annotation scope, strict exact-match evaluation, an incomplete skill dictionary, OCR errors, and spaCy false-positive organization labels. Further per-entity and per-resume analysis is required before making a final conclusion about layout-related accuracy degradation.
