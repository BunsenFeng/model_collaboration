"""
Blackboard Multi-Agent System (BBMAS) for model collaboration.

This method implements a blackboard-based multi-agent system where multiple LLMs
collaborate through a shared blackboard to solve problems iteratively.
"""
import json
import os
import random
from typing import List, Optional
import torch
from dataclasses import dataclass
from collections import defaultdict
from method import distributed_generation
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import eval


def hf_gen(model, tokenizer, prompt):
    chat = [
        {"role": "user", "content": prompt}
    ]
    chat_input = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([chat_input], return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8192, temperature=1.0)
    decoded_outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return decoded_outputs[0]


# ============================================================================
# BBMAS PROMPTS (adapted from cot-eval-harness)
# ============================================================================

GLOBAL_PROMPT_TEMPLATE = """
You are a participant in a collaborative problem-solving session. Multiple participants are working together on a shared blackboard to solve a problem. Your goal is to contribute meaningfully and help the group reach the correct solution.

**IMPORTANT**: When referring to previous work, always use the format `Entry #<entry_id>` (e.g., "Entry #1", "Entry #2"). This allows everyone to track how ideas connect and build on each other.

Below shows 1) the problem to solve and 2) the existing content on the blackboard.

---

<problem>

{problem}

</problem>

---

<blackboard_content>

{blackboard_content}

</blackboard_content>

---
""".strip()


SELECT_ACTION_PROMPT = """
Your current task is to decide how you can best contribute to solving this problem collaboratively.

**Step 1: Analyze the Current State**

Review the problem and existing blackboard content carefully:
- What has been done so far? (Reference entries using `Entry #<entry_id>`)
- What approaches have been proposed?
- Are there any errors, gaps, or unclear reasoning in existing entries?
- What still needs to be done?
- Which contribution would be most valuable right now?

Think step by step. Be critical and constructive about the existing work.

**Step 2: Select Your Action**

Based on your analysis above, choose ONE action that will best help the group:
- If you choose to *propose a solving strategy*, you should output "I choose to propose a solving strategy." here, and then stop.
- If you choose to *execute a solution step*, you should output "I choose to execute a solution step." here, and then stop.
- If you choose to *critique and refine existing work*, you should output "I choose to critique and refine existing work." here, and then stop.
- If you choose to *terminate and finalize the solution*, you should output "I choose to terminate and finalize the solution." here, and then stop.

---

**Guidelines for Action Selection**:
- Early in the process: Consider proposing a strategy or starting to execute solution steps
- If entries exist but seem unclear or potentially incorrect: Consider verification or critique
- If work is in progress: Consider executing the next step or verifying completed work
- When the solution appears complete and correct: Consider termination

---

IMPORTANT:
- Select EXACTLY ONE action from the options above
- Output the exact phrase for your chosen action and then STOP
- Do not output any other text after your action selection

---

Please respond below:
""".strip()


EXECUTE_ACTION_PROMPTS = {
    "propose a solving strategy": """At the current step, you have chosen to *{symbol}*. Below shows your reasoning process:

---

<reasoning_process>

{thinking}

</reasoning_process>

---

Now, based on your reasoning above, propose a HIGH-LEVEL STRATEGY for solving this problem.

**IMPORTANT - This is ONLY for planning, NOT solving:**
- DO NOT execute the actual solution or derive the final answer
- DO NOT work through detailed steps - save that for "execute a solution step" action
- Focus on the overall approach and roadmap

Your strategy should include:
1. **Problem Analysis**: What type of problem is this? What are the key requirements and constraints?
2. **Key Concepts**: What concepts, methods, or techniques are needed?
3. **Solution Approach**: What is the high-level method to solve this?
4. **Step Outline**: Break down the solution into 3-5 major phases/steps (without executing them)
5. **Potential Pitfalls**: What should we be careful about?

**Remember**: Reference existing entries using `Entry #<entry_id>` format.

Please output your STRATEGY (not solution) below:""",

    "execute a solution step": """At the current step, you have chosen to *{symbol}*. Below shows your reasoning process:

---

<reasoning_process>

{thinking}

</reasoning_process>

---

Now, based on your reasoning above, execute a specific step in the solution. This could include:
- Performing necessary computations or logical deductions
- Deriving intermediate results or conclusions
- Applying relevant methods, principles, or techniques
- Working through a sub-problem identified in a strategy
- Continuing work started in a previous entry

Be rigorous and show your work step-by-step.

**Remember**: Reference existing entries using `Entry #<entry_id>` format.

Please output your solution step below:""",

    "critique and refine existing work": """At the current step, you have chosen to *{symbol}*. Below shows your reasoning process:

---

<reasoning_process>

{thinking}

</reasoning_process>

---

Now, based on your reasoning above, provide constructive critique and refinements. You should:

1. **Identify target**: Specify which entry/entries you are critiquing (using `Entry #<entry_id>`)
2. **Identify issues**: Point out any errors, gaps in reasoning, unclear steps, or inefficiencies
3. **Explain concerns**: Explain why these issues are problematic
4. **Suggest improvements**: Propose specific ways to fix errors or improve the approach
5. **Offer alternatives**: If applicable, suggest alternative methods or perspectives

**Remember**: Reference existing entries using `Entry #<entry_id>` format.

Please output your critique and refinements below:""",

    "terminate and finalize the solution": """At the current step, you have chosen to *{symbol}*. Below shows your reasoning process:

---

<reasoning_process>

{thinking}

</reasoning_process>

---

Now you will finalize the solution. You should:

1. Synthesize the work from all relevant entries on the blackboard
2. Present a complete, coherent solution to the problem
3. Ensure all steps are logically connected and correct
4. State the final answer clearly

Format your output as:

FINAL:

[Your complete solution here, referencing key entries that contributed]

**Final Answer**: [State the final answer clearly]

**Remember**: Reference the entries that contributed to your solution using `Entry #<entry_id>` format.

Please output your final solution below:"""
}


VOTE_FOR_FINAL_CONCLUSION_PROMPT = """

Below shows the extracted conclusions from the existing content on the blackboard:

---

<extracted_conclusions>

{enumerated_conclusions}

</extracted_conclusions>

---

You should reason step by step and vote for the final conclusion that you think is the most correct.

Please finalize your output with: Therefore, I vote for Conclusion #<conclusion_id>.
""".strip()


# ============================================================================
# BBMAS DATA STRUCTURES
# ============================================================================

@dataclass
class Entry:
    """Blackboard entry"""
    entry_id: int
    text: str
    author_agent_id: int
    iteration: int

    def __str__(self):
        ret = ""
        ret += "---------------------\n"
        ret += f"Entry #{self.entry_id}\n"
        ret += f"Created at iteration: {self.iteration}\n"
        ret += f"Author: Agent {self.author_agent_id}\n"
        ret += "---------------------\n"
        ret += self.text + "\n\n"
        return ret


@dataclass
class Action:
    """Agent action"""
    text: str

    def is_stop(self) -> bool:
        return "I choose to terminate and finalize the solution" in self.text.strip()


class Blackboard:
    """Shared blackboard for multi-agent collaboration"""
    
    def __init__(self):
        self.user_query: Optional[str] = None
        self.entries: List[Entry] = []

    def clear(self):
        self.user_query = None
        self.entries = []

    def set_user_query(self, user_query: str):
        self.user_query = user_query

    def add_entry(self, entry: Entry):
        self.entries.append(entry)

    def get_num_entries(self) -> int:
        return len(self.entries)

    def __str__(self):
        ret = ""
        ret += "=====================\n"
        ret += "Blackboard\n"
        ret += "=====================\n"
        ret += f">>> User query:\n{self.user_query}\n\n"
        ret += ">>> Entries:\n"
        for entry in self.entries:
            ret += f"{str(entry)}\n"
        return ret


# ============================================================================
# BBMAS AGENT
# ============================================================================

class Agent:
    """Individual agent in the multi-agent system"""
    
    def __init__(
        self,
        agent_id: int,
        model_name: str,
        agent_idle_threshold: int = 6,
        max_num_retries: int = 3,
    ):
        self.agent_id = agent_id
        self.model_name = model_name
        self.agent_idle_threshold = agent_idle_threshold
        self.max_num_retries = max_num_retries
        self.last_action_step = 0

    def get_name(self):
        return f"Agent {self.agent_id} ({self.model_name})"
    
    def should_take_action(self, current_iteration: int, blackboard: Blackboard) -> bool:
        if blackboard.get_num_entries() == 0:
            return True
        if current_iteration - self.last_action_step > self.agent_idle_threshold:
            return True
        return True


# ============================================================================
# BBMAS MULTI-AGENT SYSTEM
# ============================================================================

class MultiAgentSystem:
    """Multi-agent system coordinator using distributed generation"""
    
    def __init__(
        self,
        model_names: List[str],
        shared_model: AutoModelForCausalLM,
        shared_tokenizer: AutoTokenizer,
        gpu_ids: List[int],
        max_num_agents_to_act: int = 1,
        min_num_agents_to_stop: int = 1,
        agent_idle_threshold: int = 6,
        max_iterations: int = 50,
        max_num_retries: int = 3,
    ):
        self.model_names = model_names
        self.shared_model = shared_model
        self.shared_tokenizer = shared_tokenizer
        self.gpu_ids = gpu_ids
        self.max_num_agents_to_act = max_num_agents_to_act
        self.min_num_agents_to_stop = min_num_agents_to_stop
        self.agent_idle_threshold = agent_idle_threshold
        self.max_iterations = max_iterations
        self.max_num_retries = max_num_retries

        self.blackboard = Blackboard()
        self.current_iteration = 0
        self.total_llm_calls = 0
        self._build_agents()

    def _build_agents(self):
        """Build agents"""
        self.agents = [
            Agent(
                agent_id=agent_id,
                model_name=model_name,
                agent_idle_threshold=self.agent_idle_threshold,
                max_num_retries=self.max_num_retries,
            ) for agent_id, model_name in enumerate(self.model_names)
        ]

    def _serialize_context(self, entries: List[Entry]) -> str:
        """Serialize context entries into a string"""
        if len(entries) == 0:
            return "The blackboard is currently empty."
        
        entries_by_step = defaultdict(list)
        for entry in entries:
            entries_by_step[entry.iteration].append(entry)

        ret = ""
        for iteration, iteration_entries in sorted(entries_by_step.items(), key=lambda x: x[0]):
            ret += f"\n\n---\n\nAt iteration {iteration}, the following entries were proposed:\n\n"
            for entry in iteration_entries:
                ret += f"- Entry #{entry.entry_id}:\n{entry.text}\n\n"
        return ret

    def _format_select_action_prompt(self, user_query: str, context: List[Entry]) -> str:
        """Format the action selection prompt"""
        serialized_context = self._serialize_context(context)
        full_prompt = GLOBAL_PROMPT_TEMPLATE + "\n\n" + SELECT_ACTION_PROMPT
        return full_prompt.format(problem=user_query, blackboard_content=serialized_context)

    def _format_execute_action_prompt(self, action_symbol: str, thinking: str, user_query: str, context: List[Entry]) -> str:
        """Format the action execution prompt"""
        serialized_context = self._serialize_context(context)
        execute_prompt = EXECUTE_ACTION_PROMPTS.get(action_symbol)
        if execute_prompt is None:
            return None
        full_prompt = GLOBAL_PROMPT_TEMPLATE + "\n\n" + execute_prompt
        return full_prompt.format(
            problem=user_query,
            blackboard_content=serialized_context,
            thinking=thinking,
            symbol=action_symbol
        )

    def _extract_action_symbol(self, select_action_response: str) -> Optional[str]:
        """Extract the selected action symbol from the response"""
        for symbol in EXECUTE_ACTION_PROMPTS.keys():
            if symbol in select_action_response:
                # Check no other actions are present
                other_symbols = [s for s in EXECUTE_ACTION_PROMPTS.keys() if s != symbol]
                if not any(other_symbol in select_action_response for other_symbol in other_symbols):
                    return symbol
        return None

    def _agent_run(self, agent: Agent, user_query: str, context: List[Entry]) -> Action:
        """Execute agent action using distributed generation with retry logic"""
        print(f"    → {agent.get_name()} is now selecting and executing an action...")
        select_action_prompt = self._format_select_action_prompt(user_query, context)
        
        ret = None
        for retry_idx in range(agent.max_num_retries):
            # LLM call 1: Select action
            print(f"      [Step 1/2] Calling LLM to select action (attempt {retry_idx + 1}/{agent.max_num_retries})...")
            # select_action_responses = distributed_generation.distributed_generation(
            #     [agent.model_name],
            #     [[select_action_prompt]],
            #     [gpu_id]
            # )
            # select_action_response = select_action_responses[0][0]
            select_action_response = hf_gen(self.shared_model, self.shared_tokenizer, select_action_prompt)
            self.total_llm_calls += 1
            
            # Extract action symbol
            action_symbol = self._extract_action_symbol(select_action_response)
            if action_symbol is None:
                print(f"      ⚠ Retry {retry_idx + 1}/{agent.max_num_retries}: Failed to extract action symbol from response")
                continue
            
            print(f"      ✓ Action selected: '{action_symbol}'")
            
            # Format execute action prompt
            execute_action_prompt = self._format_execute_action_prompt(
                action_symbol, select_action_response, user_query, context
            )
            if execute_action_prompt is None:
                print(f"      ⚠ Retry {retry_idx + 1}/{agent.max_num_retries}: Failed to format execute action prompt")
                continue
            
            # LLM call 2: Execute action
            print(f"      [Step 2/2] Calling LLM to execute the selected action...")
            # action_responses = distributed_generation.distributed_generation(
            #     [agent.model_name],
            #     [[execute_action_prompt]],
            #     [gpu_id]
            # )
            # action_response = action_responses[0][0]
            action_response = hf_gen(self.shared_model, self.shared_tokenizer, execute_action_prompt)
            self.total_llm_calls += 1
            
            ret = select_action_response + "\n\n---\n\n" + "To execute the action I just selected, I now generate the following content:" + "\n\n" + action_response
            
            if ret is not None:
                print(f"      ✓ {agent.get_name()} completed action successfully")
                break
        
        if ret is None:
            print(f"      ✗ {agent.get_name()} failed all retry attempts, marking as INVALID ACTION")
            ret = "INVALID ACTION"
        
        return Action(ret)

    def _agent_vote(self, agent: Agent, user_query: str, context: List[Entry], conclusions: List[str]) -> int:
        """Agent votes for final conclusion with retry logic"""
        import re
        
        print(f"    → {agent.get_name()} is voting for final conclusion...")
        serialized_context = self._serialize_context(context)
        enumerated_conclusions = "\n".join([f"Conclusion #{i+1}: {conclusion}" for i, conclusion in enumerate(conclusions)])
        
        vote_prompt_template = GLOBAL_PROMPT_TEMPLATE + "\n\n" + VOTE_FOR_FINAL_CONCLUSION_PROMPT
        vote_prompt = vote_prompt_template.format(
            problem=user_query,
            blackboard_content=serialized_context,
            enumerated_conclusions=enumerated_conclusions
        )
        
        gpu_id = self.gpu_ids[agent.agent_id % len(self.gpu_ids)]
        
        ret = None
        for retry_idx in range(agent.max_num_retries):
            print(f"      Calling LLM to vote (attempt {retry_idx + 1}/{agent.max_num_retries})...")
            # vote_responses = distributed_generation.distributed_generation(
            #     [agent.model_name],
            #     [[vote_prompt]],
            #     [gpu_id]
            # )
            # vote_response = vote_responses[0][0]
            vote_response = hf_gen(self.shared_model, self.shared_tokenizer, vote_prompt)
            self.total_llm_calls += 1
            
            # Try to extract conclusion ID from vote
            try:
                match = re.search(r'Conclusion\s*#\s*(\d+)', vote_response, re.IGNORECASE)
                if match:
                    selected_id = int(match.group(1))
                    if selected_id > 0:
                        ret = selected_id
                        print(f"      ✓ {agent.get_name()} voted for Conclusion #{ret}")
                        break
                # Fallback: try to find "Therefore, I vote for Conclusion #" pattern
                vote_prefix = "Therefore, I vote for Conclusion #"
                if vote_prefix.lower() in vote_response.lower():
                    idx = vote_response.lower().find(vote_prefix.lower())
                    remaining = vote_response[idx + len(vote_prefix):]
                    num_match = re.search(r'(\d+)', remaining)
                    if num_match:
                        selected_id = int(num_match.group(1))
                        if selected_id > 0:
                            ret = selected_id
                            print(f"      ✓ {agent.get_name()} voted for Conclusion #{ret}")
                            break
            except Exception as e:
                print(f"      ⚠ Vote attempt {retry_idx + 1}/{agent.max_num_retries}: Failed to parse vote response: {e}")
                continue
        
        if ret is None:
            ret = 1  # Default to first conclusion
            print(f"      ⚠ {agent.get_name()} failed to vote, defaulting to Conclusion #1")
        
        return ret

    def run(self, user_query: str) -> str:
        """Run the multi-agent system"""
        print("\n" + "="*80)
        print("STARTING MULTI-AGENT COLLABORATION")
        print("="*80)
        
        self.blackboard.clear()
        self.blackboard.set_user_query(user_query)
        
        while True:
            self.current_iteration += 1
            print(f"\n{'─'*80}")
            print(f"ITERATION {self.current_iteration} (Total entries on blackboard: {self.blackboard.get_num_entries()})")
            print(f"{'─'*80}")

            # Safety check: Maximum iterations reached
            if self.current_iteration > self.max_iterations:
                print(f"\n⚠ WARNING: Maximum iterations ({self.max_iterations}) reached, forcing termination...")
                break

            # Build agenda of agents that should act
            agenda = [agent for agent in self.agents if agent.should_take_action(self.current_iteration, self.blackboard)]
            
            if len(agenda) == 0:
                print("  ℹ No agents in agenda, stopping...")
                break
            
            # Select acting agents
            acting_agents = random.sample(agenda, k=min(len(agenda), self.max_num_agents_to_act))
            print(f"  📋 Agenda: {len(agenda)} agent(s) eligible to act")
            print(f"  ✓ Selected {len(acting_agents)} agent(s) to act: {', '.join([agent.get_name() for agent in acting_agents])}")
            
            # Execute actions
            print(f"\n  🔄 EXECUTING AGENT ACTIONS:")
            agent_ctx_action_tuples = []
            for acting_agent in acting_agents:
                ctx = self.blackboard.entries  # All agents see all entries
                action = self._agent_run(acting_agent, user_query, ctx)
                agent_ctx_action_tuples.append((acting_agent, ctx, action))
            
            # Apply actions to blackboard
            print(f"\n  📝 UPDATING BLACKBOARD:")
            stop_requests = 0
            for agent, ctx, action in agent_ctx_action_tuples:
                new_entry = Entry(
                    entry_id=self.blackboard.get_num_entries(),
                    text=action.text,
                    author_agent_id=agent.agent_id,
                    iteration=self.current_iteration,
                )
                self.blackboard.add_entry(new_entry)
                agent.last_action_step = self.current_iteration
                print(f"    → Added Entry #{new_entry.entry_id} by {agent.get_name()}")
                
                if action.is_stop():
                    stop_requests += 1
                    print(f"      🛑 {agent.get_name()} requested termination")

            # Termination condition
            if stop_requests >= self.min_num_agents_to_stop:
                print(f"\n{'─'*80}")
                print(f"🏁 TERMINATION CONDITION MET")
                print(f"  {stop_requests}/{len(self.agents)} agents requested stop (threshold: {self.min_num_agents_to_stop})")
                print(f"{'─'*80}")
                break
        
        print(f"\n{'='*80}")
        print(f"COLLABORATION COMPLETE")
        print(f"  Total iterations: {self.current_iteration}")
        print(f"  Total entries on blackboard: {self.blackboard.get_num_entries()}")
        print(f"  Total LLM calls: {self.total_llm_calls}")
        print(f"{'='*80}\n")

        print(self.blackboard)

        # Extract and vote on conclusions
        print(f"{'─'*80}")
        print(f"EXTRACTING AND VOTING ON CONCLUSIONS")
        print(f"{'─'*80}")
        conclusions = self._extract_conclusions()
        print(f"  Found {len(conclusions)} conclusion(s) to vote on")
        conclusion = self._select_final_conclusion(conclusions)

        return conclusion

    def _extract_conclusions(self) -> List[str]:
        """Extract final conclusions from last iteration"""
        final_entries = [entry for entry in self.blackboard.entries if entry.iteration == self.current_iteration]
        text_list = [entry.text.strip() for entry in final_entries if "FINAL:" in entry.text.strip()]
        
        if len(text_list) == 0:
            # Fallback: use all entries from last iteration
            text_list = [entry.text.strip() for entry in final_entries]
            if len(text_list) == 0:
                return ["No conclusion reached."]
            return text_list
        
        conclusions = [text.split("FINAL:")[-1].strip() for text in text_list]
        return conclusions

    def _select_final_conclusion(self, conclusions: List[str]) -> str:
        """Select final conclusion via voting"""
        if len(conclusions) == 1:
            print(f"  ℹ Only one conclusion available, selecting it by default\n")
            return conclusions[0]
        
        print(f"\n  🗳 VOTING PHASE ({len(self.agents)} agents will vote):")
        votes = []
        for agent in self.agents:
            conclusion_id = self._agent_vote(
                agent,
                self.blackboard.user_query,
                self.blackboard.entries,
                conclusions
            )
            votes.append(conclusion_id)
        
        # Aggregate votes
        vote_counts = defaultdict(int)
        for vote in votes:
            if 1 <= vote <= len(conclusions):
                vote_counts[vote] += 1
        
        if len(vote_counts) == 0:
            print(f"  ⚠ No valid votes received, defaulting to Conclusion #1\n")
            return conclusions[0]
        
        winning_conclusion_id = max(vote_counts, key=vote_counts.get)
        print(f"\n  📊 VOTING RESULTS:")
        for conclusion_id in sorted(vote_counts.keys()):
            print(f"    Conclusion #{conclusion_id}: {vote_counts[conclusion_id]} vote(s)")
        print(f"  🏆 Winner: Conclusion #{winning_conclusion_id}\n")
        
        return conclusions[winning_conclusion_id - 1]


# ============================================================================
# MODEL COLLABORATION RUN_METHOD
# ============================================================================

def run_method(task, task_type, gpu_ids, model_names, hyperparameters):
    """
    Run BBMAS method for model collaboration.
    
    Args:
        task: str, the name of the task
        task_type: str, the type of the task
        gpu_ids: list of int, GPU ids to use
        model_names: list of str, model names
        hyperparameters: dict, method-specific hyperparameters
    """

    import os
    from pathlib import Path
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent.parent.parent
    os.chdir(script_dir)
    
    # Extract method-specific hyperparameters
    max_num_agents_to_act = hyperparameters.get("max_num_agents_to_act", 1)
    min_num_agents_to_stop = hyperparameters.get("min_num_agents_to_stop", 1)
    agent_idle_threshold = hyperparameters.get("agent_idle_threshold", 6)
    max_iterations = hyperparameters.get("max_iterations", 10)
    max_num_retries = hyperparameters.get("max_num_retries", 3)
    
    print("\n" + "="*80)
    print("BBMAS (BLACKBOARD MULTI-AGENT SYSTEM) EXPERIMENT")
    print("="*80)
    print(f"📊 Task: {task} ({task_type})")
    print(f"🤖 Number of agents: {len(model_names)}")
    print(f"   Models: {', '.join(model_names)}")
    print(f"💻 GPU IDs: {gpu_ids}")
    print(f"\n⚙️  Hyperparameters:")
    print(f"   • max_num_agents_to_act: {max_num_agents_to_act}")
    print(f"   • min_num_agents_to_stop: {min_num_agents_to_stop}")
    print(f"   • agent_idle_threshold: {agent_idle_threshold}")
    print(f"   • max_iterations: {max_iterations}")
    print(f"   • max_num_retries: {max_num_retries}")
    print("="*80)

    # evaluate all given models on the dev set and select the best one to be the backbone
    dev_input_list = eval.prepare_inputs(task, task_type, "dev") # grab the inputs for the dev set

    list_of_input_list = [dev_input_list for _ in model_names] # replicate the dev inputs for each model
    list_of_output_list = distributed_generation.distributed_generation(
        model_names,
        list_of_input_list,
        gpu_ids
    ) # will be size len(model_names) x len(dev_input_list)

    list_of_dev_scores = []
    for i in range(len(model_names)):
        dev_output_list = list_of_output_list[i]
        dev_scores = eval.get_scores(task, task_type, "dev", dev_output_list)
        avg_dev_score = sum(dev_scores) / len(dev_scores)
        list_of_dev_scores.append(avg_dev_score)
        print(f"   • Model: {model_names[i]} | Avg Dev Score: {avg_dev_score:.4f}")
    
    best_model_idx = list_of_dev_scores.index(max(list_of_dev_scores))
    best_model_name = model_names[best_model_idx]
    print(f"\n🏆 Best model on dev set: {best_model_name}\n")

    # Load HF model
    # assert all(model_name == model_names[0] for model_name in model_names), "All model names must be identical"
    model_name = best_model_name
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    # Get test set inputs
    print(f"\n📥 Loading test set...")
    test_input_list = eval.prepare_inputs(task, task_type, "test")
    print(f"✓ Loaded {len(test_input_list)} test examples")
    
    # Run BBMAS for each test input
    final_output_list = []
    all_llm_calls = []
    
    for idx, test_input in enumerate(test_input_list):
        print(f"\n{'█'*80}")
        print(f"TEST EXAMPLE {idx+1}/{len(test_input_list)}")
        print(f"{'█'*80}")
        print(f"Input: {test_input[:200]}..." if len(test_input) > 200 else f"Input: {test_input}")
        
        # Create multi-agent system
        mas = MultiAgentSystem(
            model_names=model_names,
            shared_model=model,
            shared_tokenizer=tokenizer,
            gpu_ids=gpu_ids,
            max_num_agents_to_act=max_num_agents_to_act,
            min_num_agents_to_stop=min_num_agents_to_stop,
            agent_idle_threshold=agent_idle_threshold,
            max_iterations=max_iterations,
            max_num_retries=max_num_retries,
        )
        
        # Run MAS
        conclusion = mas.run(test_input)
        final_output_list.append(conclusion)
        all_llm_calls.append(mas.total_llm_calls)
        
        print(f"\n{'─'*80}")
        print(f"✅ FINAL CONCLUSION FOR EXAMPLE {idx+1}:")
        print(f"{'─'*80}")
        print(conclusion)
        print(f"\n📊 Stats: {mas.total_llm_calls} LLM calls used")
        print(f"{'█'*80}\n")
    
    # Evaluate the final outputs
    print(f"\n{'='*80}")
    print(f"EVALUATING ALL TEST EXAMPLES")
    print(f"{'='*80}")
    test_scores = eval.get_scores(task, task_type, "test", final_output_list)
    avg_test_score = sum(test_scores) / len(test_scores)
    avg_llm_calls = sum(all_llm_calls) / len(all_llm_calls)
    
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"🎯 Task: {task}")
    print(f"📊 Average test score: {avg_test_score:.4f}")
    print(f"💬 Average LLM calls per example: {avg_llm_calls:.1f}")
    print(f"📝 Total examples: {len(test_input_list)}")
    print(f"{'='*80}\n")

    # Save logs
    print(f"💾 Saving experiment logs...")
    experiment_logs = {
        "task": task,
        "task_type": task_type,
        "method": "bbmas",
        "model_names": model_names,
        "hyperparameters": hyperparameters,
        "avg_test_score": avg_test_score,
        "avg_llm_calls": avg_llm_calls,
        "logs": []
    }
    
    for i in range(len(test_input_list)):
        log_entry = {
            "input": test_input_list[i],
            "output": final_output_list[i],
            "score": test_scores[i],
            "num_llm_calls": all_llm_calls[i]
        }
        experiment_logs["logs"].append(log_entry)
    
    # Save to a json file
    log_filename = "model_collaboration/logs/{}_{}_{}_bbmas.json".format(task, len(model_names), round(avg_test_score, 4))
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    with open(log_filename, "w") as f:
        json.dump(experiment_logs, f, indent=4)
    
    print(f"✅ Results saved to: {log_filename}")
    print(f"\n{'='*80}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*80}\n")
    

if __name__ == "__main__":
    run_method()
