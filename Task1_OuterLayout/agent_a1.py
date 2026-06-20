import gc
import re
import ollama

from config import (
    TREE_CENTER,
    TREE_RADIUS,
    FRONT_SETBACK,
    REAR_SETBACK,
    LEFT_SETBACK,
    RIGHT_SETBACK,
    LayoutProposal
)

# ============================================================================
# AGENT A1
# ============================================================================

class AgentA1:

    def __init__(self):

        self.model = "llama3.2:1b"

    def generate_layout(self, feedback=None):

        gc.collect()

        prompt = f"""
You must output EXACTLY 2 lines.

FORMAT:

split_x=NUMBER
right_height=NUMBER

EXAMPLE:

split_x=58
right_height=62

RULES:
- split_x between 50 and 70
- right_height between 40 and 73
- no explanations
- no extra text

OBJECTIVE:
maximize house area while avoiding tree collision.

TREE:
center={TREE_CENTER}
radius={TREE_RADIUS}

SETBACKS:
front={FRONT_SETBACK}
rear={REAR_SETBACK}
left={LEFT_SETBACK}
right={RIGHT_SETBACK}

FEEDBACK:
{feedback if feedback else "None"}
"""

        response = ollama.chat(

            model=self.model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            options={

                "temperature": 0.1,

                "num_predict": 20
            }
        )

        text = response["message"]["content"].strip()

        print("\nRAW LLM OUTPUT:")
        print(text)

        # =====================================================
        # FLEXIBLE PARSER
        # =====================================================

        split_match = re.search(
            r'split_x\s*=?\s*(\d+\.?\d*)',
            text,
            re.IGNORECASE
        )

        height_match = re.search(
            r'right_height\s*=?\s*(\d+\.?\d*)',
            text,
            re.IGNORECASE
        )

        # =====================================================
        # FALLBACK
        # =====================================================

        if split_match and height_match:

            split_x = float(
                split_match.group(1)
            )

            right_height = float(
                height_match.group(1)
            )

        else:

            nums = re.findall(
                r'\d+\.?\d*',
                text
            )

            if len(nums) >= 2:

                split_x = float(nums[0])

                right_height = float(nums[1])

            else:

                raise Exception(
                    f"Could not parse LLM output:\n{text}"
                )

        # =====================================================
        # CLAMP VALUES
        # =====================================================

        split_x = max(
            50,
            min(70, split_x)
        )

        right_height = max(
            40,
            min(73, right_height)
        )

        return LayoutProposal(
            split_x,
            right_height
        )
