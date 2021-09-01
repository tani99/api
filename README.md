# Flask API for EduNote

## Contents

This repository contains the python code required for textual processing for this thesis. The Flutter frontend makes http requests to a locally hosted version of this application to process textual information.

## Abstract

This thesis aims to create an automated note-making software that can process educational texts and generate quality study materials they provide. This work is inspired by the need to assist teachers in low-income countries to improve the quality of study materials. The subject area of the input educational texts is history and civics. The task is broken down into three fundamental problems to tackle this, i.e. summarization, simplification and visualization.


Summarization is implemented inspired by Edmundson's extractive summarization method. We introduce and incorporate a new Length method for Edmundson's algorithm, which influences the quality of extractive summarization on educational texts. 

Text summarization is combined with simplification techniques to improve generated text quality under the assumption that simpler text forms superior study materials. We explore two simplification methods and identify the best one as a simplification system by Facebook research; Multilingual Unsupervised Sentence Simplification by Mining Paraphrases (MUSS). We introduce a new approach of preprocessing input data provided to the MUSS system in pairs of sentences. This improves performance by finetuning the system to merge pairs of sentences that improve the concise representation of information.

For visualization, we select two visual representations of text, specific to the chosen domain, i.e., tables and timelines. Tables represent information from input texts that contain comparable data. Timelines represent input text that detail information that follows a progression of dates. An iterative approach to text segmentation is implemented to breakdown texts into sub-components. We improve this approach by introducing an improved similarity metric that results in superior segmentation for our educational data. These sub-components are processed and represented in the desired visualization forms. Additionally, we use NLP techniques to recognize and highlight important terms in the generated text and improve the structural representation of the information through bullet points.

The textual processing and visualization research conducted comes together in the form of the developed software application, EduNote. This application allows users to generate automated summaries using the researched techniques in a user-friendly manner. It allows users to edit and improve the automated summaries and is developed as an assistive software that speeds up the note-making process.
