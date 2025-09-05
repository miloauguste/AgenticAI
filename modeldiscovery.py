from transformers import pipeline
from transformers.pipelines import SUPPORTED_TASKS

def discover_all_defaults():
    for task_name in SUPPORTED_TASKS.keys():
        try:
            pipe = pipeline(task_name)
            model = pipe.model.config.name_or_path
            print(f"📝 {task_name}: {model}")
        except Exception as e:
            print(f"❌ {task_name}: Error - {str(e)[:50]}...")

discover_all_defaults()