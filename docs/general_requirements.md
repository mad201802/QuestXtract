# General Requirements
**Project**: QuestXtract - a Python package for extracting questions from lecture audio and video.

**Team:** 5 members · **Version:** 1.0 · **Date:** 07.09.2026
## 1. Purpose & Scope
The project delivers an installable Python package that takes a lecture recording (preferably video) and returns the questions spoken during it in a structured, machine-readable form. Based on that, generate quiz questions for the lecture including blackboard writings / animations, etc. And provide a set of answers for the questions based on the content. 

**Out of scope:** speaker identification by name

## 2. Functional Requirements
1. The system shall accept common video data (e.g. .mp4)
2. The system shall extract the audio track from video input.
3. The system shall transcribe the audio to text using an automatic speech recognition model.
4. The system shall extract animation/drawing from the video input.
5. The system shall let a user/teacher edit the questions provided by the system after extraction. 
6. The system shall identify question sentences within the transcript.
7. The system shall return each question with its start timestamp in the recording.
8. The system shall export results to a structured file format (JSON).
9. The system shall report progress for long recordings.
10. The system shall show the user how confident it is on the detection
11. (The system shall have a type of filtering based on timestamps or topics)

## 3. Non-Functional Requirements
1. The package shall run on Python >=3.10 under Linux, macOS and Windows.
2. Question detection shall reach a recall of at least 50% on the team's manually labelled test set.
3. Public API functions shall have docstrings, type hints, and unit tests; overall test coverage ≥ 70%.
4. The package shall be tested on every pull request before merge into main (pipeline)

## 4. Constraints & Assumptions
1. Implementation language: Python
2. Recordings are assumed to be in English and of reasonable audio quality (single main speaker, limited noise).
3. Development team of 5, working part-time alongside coursework; total duration ~ 10 weeks. With 2 weeks sprints 
4. Test material: Any lecture video from youtube. 
5. The model shall not create questions on their own, only extract from questions asked. 

## 5. Milestones
The customer's phases, as agreed. The team plans and tracks them as [GitHub milestones](https://github.com/mad201802/QuestXtract/milestones), which are the reference for scope and due dates.

1. Get familiar with the current audio only codebase and refactor it
2. Deployable in own machine or in server -> github repo with a python package
3. Get information from the images considering them as static, and enrich the audio information
4. Get information from the images in a flow based way and enrich the audio information
Make manim animations out of the images flow, [manim](https://github.com/manimCommunity/manim)

## 6. Acceptance Criteria
The project is accepted when a fresh install of the package processes an unseen lecture recording end-to-end and produces a JSON file of timestamped questions, all Must requirements are demonstrated in the final presentation, and the test suite passes in CI.
