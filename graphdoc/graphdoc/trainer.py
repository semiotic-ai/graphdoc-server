# system packages
import os
import json
import asyncio
from pathlib import Path

# internal packages
from .executor import OpenAILanguageModel, EntityComparisonPromptExecutor
from .prompt import Prompt

# external packages
from dotenv import load_dotenv

async def main():
    print("hello, world!")

    # set up required classes
    load_dotenv(".env")
    lm = OpenAILanguageModel(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    ecpe = EntityComparisonPromptExecutor(
        language_model=lm,
    )

    # get the test set
    assets_dir = Path(__file__).parent.parent.parent / "assets" / "tests"
    assets_dir = Path(assets_dir)
    if not assets_dir.exists():
        raise FileNotFoundError(f"assets directory not found at: {assets_dir}")

    with open(Path(assets_dir / "entity_comparison_assets.json"), "r") as f:
        entity_comparison_assets = json.load(f)

    gold_entity_comparison = "".join(entity_comparison_assets["gold"]["prompt"])
    four_entity_comparison = "".join(entity_comparison_assets["four"]["prompt"])
    three_entity_comparison = "".join(entity_comparison_assets["three"]["prompt"])
    two_entity_comparison = "".join(entity_comparison_assets["two"]["prompt"])
    one_entity_comparison = "".join(entity_comparison_assets["one"]["prompt"])
    test_assets = {
        "gold_entity_comparison": gold_entity_comparison,
        "four_entity_comparison": four_entity_comparison,
        "three_entity_comparison": three_entity_comparison,
        "two_entity_comparison": two_entity_comparison,
        "one_entity_comparison": one_entity_comparison,
    }

    # set the base prompt
    prompt = Prompt(
        title="Entity comparison prompt",
        base_content=ecpe.instantiate_prompt(
            template_name="entity_comparison_prompt.txt",
            template_variables={
                "entity_pred": "tests",
                "entity_gold": "tests",
            },
        ),
        metadata={"score": 0},
    )

    # temperature range to check
    temps = [0.6, 0.7, 0.8]

    for i in range(3):
        template_iteration_name = f"entity_comparison_prompt_{i}.txt"

        print(f"Iteration: {i}")
        score = 0

        four_comparison = None
        three_comparison = None
        two_comparison = None
        one_comparison = None

        for t in temps:
            tasks = [
                asyncio.to_thread(
                    ecpe.execute_prompt,
                    template_name=template_iteration_name,
                    template_variables={
                        "entity_pred": test_assets["gold_entity_comparison"],
                        "entity_gold": test_asset,
                    },
                )
                for test_asset in test_assets
            ]

            test_asset_comparisons = await asyncio.gather(*tasks)
            try:
                parsed_test_asset_comparisons = [
                    ecpe.language_model.parse_response(r)
                    for r in test_asset_comparisons
                ]
            except:
                continue

            four_comparison_score = abs(
                4 - parsed_test_asset_comparisons[0]["correctness"]
            )
            three_comparison_score = abs(
                3 - parsed_test_asset_comparisons[1]["correctness"]
            )
            two_comparison_score = abs(
                2 - parsed_test_asset_comparisons[2]["correctness"]
            )
            one_comparison_score = abs(
                1 - parsed_test_asset_comparisons[3]["correctness"]
            )

            if four_comparison_score != 0:
                four_comparison = parsed_test_asset_comparisons[0]
            if three_comparison_score != 0:
                three_comparison = parsed_test_asset_comparisons[1]
            if two_comparison_score != 0:
                two_comparison = parsed_test_asset_comparisons[2]
            if one_comparison_score != 0:
                one_comparison = parsed_test_asset_comparisons[3]

            score += (
                four_comparison_score
                + three_comparison_score
                + two_comparison_score
                + one_comparison_score
            ) / 4
            print(f"updated score: {score}")
        iteration_score = score / 3
        print(f"iteration score: {iteration_score}")

        revised_prompt = ecpe.execute_four_comparison_prompt(
            original_prompt_template=ecpe.get_prompt_template(
                template_name=template_iteration_name,
            ).render({"entity_pred": "entity_pred", "entity_gold": "entity_gold"}),
            four_comparison=four_comparison,
            three_comparison=three_comparison,
            two_comparison=two_comparison,
            one_comparison=one_comparison,
        )
        try:
            parsed_revised_prompt = ecpe.language_model.parse_response(revised_prompt)
        except:
            try:
                revised_prompt = ecpe.execute_four_comparison_prompt(
                    original_prompt_template=ecpe.get_prompt_template(
                        template_name=template_iteration_name,
                    ).render(
                        {"entity_pred": "entity_pred", "entity_gold": "entity_gold"}
                    ),
                    four_comparison=four_comparison,
                    three_comparison=three_comparison,
                    two_comparison=two_comparison,
                    one_comparison=one_comparison,
                )
                parsed_revised_prompt = ecpe.language_model.parse_response(
                    revised_prompt
                )
            except:
                continue
        formatted_revised_prompt = ecpe.format_entity_comparison_revision_prompt(
            revised_prompt
        )
        print("new prompt written to file")
        new_file = (
            Path(ecpe.prompt_templates_dir_path)
            / f"entity_comparison_prompt_{i + 1}.txt"
        )
        with open(new_file, "w") as file:
            file.write(formatted_revised_prompt)

    # for a given prompt, against each temperature, get the abs dif from the gold and pred and average that value
    # then, iterate the prompt and try again
    # do this for a set number of iterations


if __name__ == "__main__":
    asyncio.run(main())
