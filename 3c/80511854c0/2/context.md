# Session Context

## User Prompts

### Prompt 1

I want you to double check end to end that all the branches are fully complete and work totally. Then go ahead and merge each one to main. When you are finished, we'll use the smallest available coding model from HuggingFace and we'll try to run our system on a small coding problem.

### Prompt 2

Go for it

### Prompt 3

Let's try something more complicated. Like a web-app or some other full project.

### Prompt 4

[Request interrupted by user]

### Prompt 5

Wait, there should be no fastapi any more. Remember all the work we just did? /Users/noahdolevelixir/Code/rune/instructions/swarm-execution-proposal-v2.md

### Prompt 6

[Request interrupted by user]

### Prompt 7

Wait, we should be generating tasks which are handled by the swarm to complete the overall project we defined in our test. The goal of this test is to check that our whole implementation works as expected.

### Prompt 8

[Request interrupted by user]

### Prompt 9

Not exactly. We want to define a project to our app (using our hypernetwork approach to give it info on the project details) and instruct it to break the project into tasks handled by the swarm where we iterate using the hypernetwork diff trajectory approached the project was defined for.

### Prompt 10

<task-notification>
<task-id>a7ead76618e2531a1</task-id>
<tool-use-id>toolu_01WijCXfLDAxo9ocWVWAj9U8</tool-use-id>
<output-file>/private/tmp/claude-503/-Users-noahdolevelixir-Code-rune/tasks/a7ead76618e2531a1.output</output-file>
<status>completed</status>
<summary>Agent "Explore hypernetwork and trajectory system" completed</summary>
<result>Now I have a comprehensive understanding of the Rune codebase. Let me provide a detailed analysis:

## Complete Rune Hypernetwork / Parametric Memory / ...

### Prompt 11

This still isn't 100% correct. The first attempt uses the hypernetwork to pass the general method for planning, executing, coding and validating to the model. The prompt is just used to launch the process. The first prompt could be the project definition. This prompt would cause us to generate an implementation plan which would be passed to the next iteration via hypernetwork to preserve context window.

