<div align="center">
  <img src="img/logo.png" alt="Ground-Up Machine Learning" width="400"/>
  <p><strong>Fast-Track to Machine Learning: A Curriculum Crafted for Newbies and Busy Bees</strong></p>
  <!-- Badges -->
  <img alt="GitHub License" src="https://img.shields.io/github/license/bedirt/Ground-Up-Machine-Learning">
  <!-- <img alt="Static Badge" src="https://img.shields.io/badge/Completed-1%2F4-FC8268?style=flat&labelColor=193A4C"> -->
  <img alt="Static Badge" src="https://img.shields.io/badge/Completed-1%2F4-60DDA0?style=flat&labelColor=193A4C">
</div>

Welcome to Ground-Up Machine Learning (GML) - a personal project that sprang from my desire to demystify the world of machine learning (ML) for friends and colleagues who expressed a keen interest but found themselves overwhelmed by the existing resources. This course is my attempt to break down ML into digestible, engaging, and accessible segments, ensuring that anyone, regardless of their background, can grasp the fundamentals of machine learning and see its beauty and utility in the modern world.

## Table of Contents

1. [Introduction & Overview](#introduction--overview)
   - [The Genesis of GML](#the-genesis-of-gml)
   - [What We Hope to Achieve](#what-we-hope-to-achieve)
   - [Target Audience](#target-audience)
   - [Prerequisites](#prerequisites)
   - [Tools and Libraries](#tools-and-libraries)
   - [Course Structure](#course-structure)
   - [Expected Effort](#expected-effort)
   - [Ideal Way to Follow the Curriculum](#ideal-way-to-follow-the-curriculum)
   - [Join the Journey](#join-the-journey)
2. [Week 1: Introduction to ML & Supervised Learning - Regression](#week-1-introduction-to-ml--supervised-learning---regression)
   - [Lecture Table of Contents](#lecture-table-of-contents)
   - [Week 1 Homework](#week-1-homework)


## Introduction & Overview

### The Genesis of GML

The inception of GML was not a planned endeavor but a response to the genuine curiosity and, at times, frustration voiced by those around me. They sought a pathway into ML that didn't start with daunting mathematics or require sifting through countless hours of content to find practical, engaging learning experiences. As someone deeply passionate about making education accessible and enjoyable, I saw an opportunity to create something unique—a course that would serve as an introductory handshake to ML, offering a friendly welcome to beginners and a clear, concise overview for busy learners. Thus, GML was born, crafted from the ground up to introduce you to the marvels of machines that learn.

### What We Hope to Achieve

GML is designed to light a spark of interest in machine learning, providing you with:
- A foundational understanding of ML concepts applied in real-world scenarios.
- The ability to distinguish between different types of ML: Supervised, Unsupervised, and Reinforcement Learning.
- Practical skills through hands-on experience with essential ML tools and libraries, applying your knowledge to real datasets.

### Target Audience

This course is ideally suited for:
- Absolute beginners curious about machine learning.
- Busy individuals seeking a streamlined yet meaningful overview of ML.
- Educators looking for a structured guide to introduce ML concepts to students.

### Prerequisites

A foundational knowledge of Python is expected, as the course will briefly touch upon Python basics but primarily focus on applying it within ML contexts. Additionally, while not math-heavy, a basic understanding of calculus and linear algebra can enrich your learning experience. Nonetheless, these are not dealbreakers; with a willingness to learn and explore these topics on your own, you'll find this course both manageable and rewarding.

### Tools and Libraries

We'll be leveraging several key Python tools and libraries throughout GML:
- **Python:** The primary language of instruction.
- **NumPy and Pandas:** For numerical computing and data manipulation.
- **Matplotlib/Seaborn:** For visualizing data.
- **Scikit-learn:** For easy-to-use ML models.

### Course Structure

GML unfolds over four weeks, each dedicated to a different facet of machine learning:
1. **Introduction to ML & Supervised Learning - Regression**
2. **Supervised Learning - Classification**
3. **Unsupervised Learning & Clustering**
4. **Introduction to Reinforcement Learning**

Each week includes a lecture guide and lab materials designed not as exhaustive resources but as interactive guides akin to slides that a teacher can use to lead discussions, demonstrations, and hands-on projects. The course is crafted with the classroom in mind, requiring an instructor to breathe life into the content. As such, it's perfect for educators or study groups seeking a structured path to explore ML together.

### Expected Effort

To fully benefit from what GML offers, participants are expected to commit to approximately 2-3 hours of lecture time and 6-7 hours of self-study each week. This self-study time includes going through lab materials, completing assignments, and additional reading or practice as needed. The weekly structure is flexible and can be adjusted based on individual or group needs, making the course adaptable to different learning environments.

### Ideal Way to Follow the Curriculum

GML is structured to maximize learning through interaction, hands-on practice, and experimentation. To get the most out of this course, we recommend the following approach:

1. **Engage with the Lecture Content:** Whether you're learning solo with an online community, in a classroom, or part of a study group, start by digesting the lecture content together. This collaborative approach allows for discussion, clarification of concepts, and sharing of insights, making the learning experience richer and more comprehensive.

2. **Hands-on Practice with Lab Material:** After the lecture, dive into the lab material. This is where you'll get practical experience with the topics and methods introduced during the lecture. The labs are designed to be interactive, allowing you to apply what you've learned in a guided environment. Ideally, this should be done with the support of a teacher or within your study group, providing a collaborative space to explore and learn from each other.

3. **Weekly Assignments:** Each week, you'll be tasked with an assignment that encourages you to implement and experiment with the week's topics. These assignments are crucial for deepening your understanding, as they require you to build concepts from scratch. Writing and debugging your code, and reflecting on your approach, will ensure you have a solid grasp of the material. These assignments are not only about reinforcing what you've learned but also about fostering creativity and problem-solving skills in real-world scenarios.

#### Why This Approach?

Learning machine learning, or any complex subject, is best achieved through a blend of theoretical understanding and practical application. By following this structured approach, you're not just passively absorbing information; you're actively engaging with the material, applying it in practical contexts, and solidifying your understanding through creation and collaboration. This methodology is designed to cater to diverse learning styles and to accommodate varying schedules, making ML accessible to everyone interested in embarking on this journey.

### Join the Journey

GML is more than just a course; it's an invitation to explore the world of machine learning in a way that's engaging, accessible, and, most importantly, grounded in real-world application. Whether you're teaching a class, learning with peers, or guiding yourself through the fundamentals of ML, GML offers the tools, insights, and inspiration to embark on this exciting journey.

Welcome to Ground-Up Machine Learning. Let's discover the power of machines that learn, together.

## Week 1: Introduction to ML & Supervised Learning - Regression

In this week, we will start with the basics of Machine Learning, and then move on to the fundamentals of Supervised Learning, focusing on Regression. We will also introduce the tools we will be using throughout the course, and discuss the importance of understanding data in Machine Learning.

I found that the best way to learn is by doing, so I am emphasizing that when teaching/studying this course we need to ask bunch of questions and do a lot of discussions. Also in the notebooks I tried to add bunch of interactive examples to make the learning process more fun and engaging.

### Lecture Table of Contents

<table border="1">
  <tr>
    <th>Topic</th>
    <th>Details</th>
    <th>Resources: Lecture Notebook / Lab Notebook</th>
  </tr>
  <tr>
    <td><strong>Introduction to Machine Learning and Its Evolution</strong></td>
    <td>
      <ul>
        <li><strong>History and Evolution of AI and Machine Learning</strong>: Quick overview, from perceptrons to deep learning.</li>
        <li><strong>Distinction between ML/AI/DS/DL</strong>: Clarifying the terms Artificial Intelligence (AI), Machine Learning (ML), Deep Learning (DL), and Data Science (DS), and how they relate to each other.</li>
        <li><strong>Types of ML</strong>: Explaining the different Machine Learning methods: Supervised Learning, Unsupervised Learning and Reinforcement Learning.</li>
      </ul>
    </td>
    <td rowspan="4">
      <ul>
        <li>Lecture Notebook: <a href="week_1/GML_Lecture_1.ipynb">GML_Lecture_1.ipynb</a></li>
        <li>Lab Notebook: <a href="week_1/GML_Lab_1.ipynb">GML_Lab_1.ipynb</a></li>
        <li>Lab Dataset: <a href="week_1/customer_spending_behaviour.csv">customer_spending_behaviour.csv</a></li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><strong>Introcudtion to the Tools We will Utilize</strong></td>
    <td>
      <ul>
        <li><strong>Interactive Notebook</strong>: Showing how we are making use of interactive python notebooks, and why are they a good choice for us, and how to use them.</li>
        <li><strong>Python</strong>: Very quick and brief reminder of some python components.</li>
        <li><strong>Numpy</strong>: Mentioning the basic functions of numpy and how to use it.</li>
        <li><strong>Pandas</strong>: Tad bit of this as well!</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><strong>Understanding Data in Machine Learning</strong></td>
    <td>
      <ul>
        <li><strong>Features and Targets</strong>: Introduction to the concepts of features (independent variables) and targets (dependent variables).</li>
        <li><strong>Data Visualization</strong>: Demonstrating the use of plots to explore relationships in data.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td><strong>Fundamentals of Supervised Learning: Regression</strong></td>
    <td>
      <ul>
        <li><strong>Introduction to Supervised Learning</strong>: Basic introduction to what supervised learning is, and specifically regression.</li>
        <li><strong>Real-world examples</strong>: Discussing some real world scenarios and how regression is used in the real world.</li>
        <li><strong>Classification vs. Regression</strong>: Highlighting the differences, focusing on the predictive nature of regression for continuous outcomes.</li>
        <li><strong>Linear Regression Basics</strong>: Introduction to the simplest form of predictive modeling.</li>
        <li><strong>Loss Functions</strong>: Discussion on different ways to measure model accuracy, emphasizing Mean Squared Error (MSE) and Mean Absolute Error (MAE) for their simplicity and interpretability.</li>
        <li><strong>Overfitting and Underfitting</strong>: Concepts of overfitting and underfitting covered. How model complexity effects overfitting and underfitting.</li>
        <li><strong>Train-Test-Val Split</strong>: Importance of splitting data into training, test and validation sets to evaluate model performance realistically.</li>
      </ul>
    </td>
  </tr>
</table>

### Week 1 Homework

**Task(s):**
- Ground up coding of the linear regression model and all the bits and pieces we discussed in class using only python and numpy.
- Using this model you built to train a regression model for a dataset you select. Use https://www.kaggle.com/datasets to find a dataset and a question you want to answer!
- Then using your model, and some data cleaning answer your question.
- Follow all the concepts you learned in class (and maybe even more) to solidify what we learned!

### Contributions

We welcome contributions to the Ground-Up Machine Learning (GML) curriculum! Whether you're interested in adding new content, suggesting improvements, or fixing bugs, your input is valuable in making this an even better resource for everyone.

To contribute, please follow these steps:
1. **Fork the Repository:** Start by forking the GML repository to your own GitHub account.
2. **Make Your Changes:** Whether it's adding new materials, correcting typos, or suggesting enhancements, make your changes in your forked version.
3. **Submit a Pull Request:** Once you're happy with your updates, submit a pull request back to the main GML repository. Please provide a clear description of your changes and the reasons for them.
4. **Review Process:** Your pull request will be reviewed by me or the maintainers of the repo. We may engage with you for discussions or request modifications before merging your contributions.

For significant changes or new content, we recommend opening an issue to discuss your ideas with us before proceeding. This collaborative approach ensures that we maintain the integrity and coherence of the curriculum while incorporating the community's valuable insights.

### License

The Ground-Up Machine Learning (GML) course is made available under the [GNU General Public License v3.0](LICENSE). You are free to use, share, and modify the course materials for educational purposes, provided you adhere to the terms of the license.

Please review the full license for more details. This open license is part of our commitment to supporting and contributing to the open-source community, making learning accessible to as many people as possible.
