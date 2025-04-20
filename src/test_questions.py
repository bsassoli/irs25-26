TEST_QUESTIONS = [
        "Who was Keeling and what is his relevance?",
        "What is the observer-expectancy effect?"
        "What are thought experiments in scientific reasoning?",
        "What was the Reber plan?"
        "What is a meta-analysis?",
        "What are the pitfalls of meta-analyses?",
        "What are Lagrange points?",
    ]


TEST_ANSWERS = [
"""
Charles David Keeling was a geochemist who played a pivotal role in climate science through his groundbreaking measurements of atmospheric carbon dioxide (CO₂). In 1958, Keeling installed infrared gas analyzers at the Mauna Loa Observatory in Hawai‘i, where he began systematically recording CO₂ levels in the atmosphere. These continuous measurements created what is now famously known as the Keeling Curve​.

The Keeling Curve is significant for two main reasons:

Empirical Evidence of Rising CO₂: It was the first clear, systematic evidence that atmospheric CO₂ levels were steadily increasing over time. This increase was linked directly to human activities, such as the burning of fossil fuels.

Foundation for Climate Change Science: Keeling's data provided the baseline for understanding the greenhouse effect's intensification due to anthropogenic (human-caused) emissions. His work directly contributed to the broader scientific consensus that human actions are driving global climate change.

Keeling’s contributions are foundational because they offered hard, quantifiable data confirming that the Earth's atmospheric composition was changing at an unprecedented rate. His research helped to shift climate science from theoretical speculation to observable, measurable reality.
""",
"""
The observer-expectancy effect is a cognitive bias in scientific research where a researcher's expectations inadvertently influence the behavior of participants, potentially skewing the results. This influence can be unconscious and subtle, often arising through body language, facial expressions, or changes in tone that subjects may pick up on.

A well-known historical example is the case of Clever Hans, a horse in early 20th-century Germany that seemed capable of solving arithmetic problems. Investigations eventually revealed that Hans was responding not to the math problems themselves, but to involuntary cues given by his human questioners. When those cues were hidden or eliminated—such as when the questioner did not know the answer or was positioned behind a screen—Hans’s accuracy dramatically declined​.

In modern experimental settings, this effect is taken seriously and controlled for using techniques such as blinding. In a double-blind experiment, neither the subjects nor the researchers know who is receiving the experimental treatment, which helps prevent expectancy effects from influencing results​.
""",
"""
Thought experiments play a crucial role in scientific reasoning by allowing researchers to explore hypotheses and implications through imagination rather than physical intervention. They involve imagined interventions on imagined systems to better understand how an independent variable might influence a dependent variable in the real world​.

Functions of Thought Experiments
Hypothesis Testing: They help assess the plausibility of a hypothesis when direct experimentation isn't feasible, practical, or ethical.

Revealing Conceptual Problems: They can expose contradictions or problems in existing theories or intuitions, prompting revision or clarification.

Illustrating Theories: They often clarify complex theoretical points or provide intuitive grounding for abstract concepts.

Classic Examples
Galileo's falling bodies: Galileo imagined dropping a light and heavy object tied together, showing logically that the idea heavier objects fall faster is flawed​.

Newton’s cannon: Newton asked readers to imagine firing a cannon from a mountain to illustrate orbital motion and gravitational force.

Schrödinger’s cat: Schrödinger used this thought experiment to highlight the strangeness of quantum superposition and the measurement problem, not to discredit quantum mechanics but to demand a better interpretation of its implications​.

Limitations
Bounded by Imagination: Our ability to conceive all relevant factors or implications may be limited—particularly for phenomena far outside everyday experience.

Risk of Misleading Intuition: Since they rely on intuition, thought experiments may lead to erroneous conclusions if those intuitions are flawed.

Summary
Thought experiments are especially valuable in foundational science—like physics and philosophy of science—because they enable critical insight without requiring material resources. They are akin to 'dry runs' for ideas, offering a space to test, challenge, and refine conceptual frameworks​.
""",
"""
The Reber Plan was a proposal in the 1940s by John Reber, a theatrical producer with no scientific or engineering background, to dramatically reshape the San Francisco Bay. Reber believed that the bay was a "geographic mistake" interfering with industrial development in San Francisco.

His plan involved:

Filling in parts of the bay to create additional land for industrial use such as airports and factories.

Constructing dams to create two freshwater lakes from the rivers flowing into the bay.

Using these lakes to provide drinking water and irrigation to the region​.

Although Reber's proposal was initially taken seriously by some politicians and business leaders, the U.S. Army Corps of Engineers tested the plan using the Bay Model, a hydraulic model of the bay. The results showed that implementing the Reber Plan would lead to disastrous consequences, such as stagnant, unhealthy water and destructive currents. Based on this modeling, the plan was abandoned​.
""",
"""
A meta-analysis is a research method used to combine the results of multiple existing studies—typically experiments or observational studies—that address the same hypothesis or question. The aim is to strengthen the conclusions that can be drawn by integrating findings across studies, especially when individual studies have limitations or show conflicting results.

In practice, researchers conducting a meta-analysis:

Identify a research question.

Search for all relevant studies on that question.

Select studies to include, based on specific criteria.

Calculate an effect size for each study (a measure of the strength of a phenomenon).

Use statistical techniques to combine these effect sizes into an overall estimate.

Analyze variation across studies to identify potential causes of discrepancies.

This approach helps improve both internal validity (by accounting for confounding variables) and external validity (through broader sampling of circumstances and populations).

An example from the text involves a meta-analysis of the placebo effect, in which researchers analyzed 11 studies out of an initial pool of 1,246. They found a large overall effect size for placebos, even when patients knew they were receiving them, though with significant variation across conditions such as back pain, depression, and allergies​.

Meta-analyses are especially useful in fields like healthcare, where individual studies may be small or inconsistent, but they are increasingly used in other areas of science as well​.
""",
"""
Meta-analyses are powerful tools for synthesizing scientific knowledge, but they also come with several pitfalls and limitations, as discussed in Chapter 4 of Recipes for Science. Here are some key concerns:

🔻 1. Low Quality or Biased Input Studies
“A meta-analysis is only as good as the studies it includes.”
If many of the included studies are poorly designed or biased, then the meta-analysis may simply reinforce those flaws, rather than correct them​.

🔻 2. Selection Bias
Researchers must choose which studies to include in the meta-analysis. These decisions—often based on factors like study design, outcome reporting, or sample characteristics—can introduce bias if not handled transparently and systematically​.

🔻 3. Apples-to-Oranges Problem
Combining results from studies that are too different in terms of methods, populations, or contexts can be misleading:

For example, a placebo might show effectiveness for back pain but not for depression. If lumped together, the overall effect size might not reflect the truth for any one condition​.

🔻 4. Masking Important Variation
A meta-analysis produces an average effect size, which might obscure important differences across subgroups or study contexts. This could lead to overgeneralizations or incorrect conclusions about the reliability or magnitude of an effect​.

🔻 5. Publication Bias
Studies with statistically significant results are more likely to be published. Meta-analyses that include only published studies may overestimate effect sizes and miss null or contradictory findings, skewing the overall conclusions​.

In sum, while meta-analysis is a valuable method for gaining insights from multiple studies, its reliability depends critically on the quality of included research, the consistency of methods, and transparency in selection and analysis procedures.
""",
"""
Lagrange points are specific solutions to a special case of what is known in physics as the three-body problem—the challenge of predicting the motion of three celestial bodies based on their positions, momenta, and gravitational interactions. This problem is famously difficult and lacks a general solution, but certain specific configurations have been solved.

In particular, when one of the three bodies is nearly massless relative to the other two (e.g., a spacecraft in the Earth-Sun system), five unique solutions emerge. These five positions are called Lagrange points, named after Joseph-Louis Lagrange, who, along with his teacher Leonhard Euler, worked on these problems in the 19th century.

Definition:
Lagrange points are regions in space where the gravitational forces of two large orbiting bodies (like the Earth and the Sun) precisely balance the centripetal force needed for a much smaller third body (like a spacecraft) to move with them. In other words, a spacecraft placed at one of these points can remain in a stable position relative to the two larger bodies, requiring minimal energy to maintain its location​.

Configuration:

The first three points (L1, L2, L3) are located along the line connecting the two massive bodies.

The last two points (L4 and L5) form the apexes of equilateral triangles with the two massive bodies at the base. This overall configuration resembles a “peace” sign​.

Example:
The James Webb Space Telescope (JWST) was sent to the L2 point of the Sun-Earth system. This location—about 1.5 million kilometers from Earth—allows the telescope to maintain a stable orbit with minimal fuel usage, harvest solar energy, and maintain clear lines of communication with Earth. However, the stability is not perfect, and small deviations from equilibrium can grow over time, requiring periodic course corrections​.

In sum, Lagrange points exemplify a powerful application of deductive modeling in science, using mathematical reasoning to solve physical problems about motion and gravitational dynamics.
"""
]