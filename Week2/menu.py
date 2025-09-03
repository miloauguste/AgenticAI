import gradio as gr
import json

# Sample data for dropdowns
MODELS = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
    "Google": ["gemini-pro", "gemini-1.5-pro", "palm-2"]
}
MODELS2 = {
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "Anthropic": ["claude-3-sonnet", "claude-3-haiku", "claude-3-opus"],
    "Google": ["gemini-pro", "gemini-1.5-pro", "palm-2"]
}

DESTINATIONS = ["London", "Paris", "Tokyo", "Berlin", "New York"]
LANGUAGES = ["English", "Spanish", "French", "German", "Japanese"]
PRIORITIES = ["Economy", "Business", "First Class"]


# =============================================================================
# EXAMPLE 1: INDEPENDENT DROPDOWNS (Simple)
# =============================================================================

def example1_simple_dropdowns():
    """Multiple independent dropdowns - simplest approach"""

    def process_selections(model_provider, destination, language, priority):
        result = f"""
        Selected Options:
        - Model Provider: {model_provider}
        - Destination: {destination} 
        - Language: {language}
        - Priority: {priority}
        """
        return result

    with gr.Blocks(title="Example 1: Independent Dropdowns") as demo:
        gr.Markdown("# Independent Dropdowns Example")

        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                label="Model Provider",
                value="OpenAI"
            )
            destination_dropdown = gr.Dropdown(
                choices=DESTINATIONS,
                label="Destination",
                value="London"
            )

        with gr.Row():
            language_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                label="Response Language",
                value="English"
            )
            priority_dropdown = gr.Dropdown(
                choices=PRIORITIES,
                label="Service Priority",
                value="Economy"
            )

        output = gr.Textbox(label="Current Selections", lines=6)
        submit_btn = gr.Button("Update Selections")

        # Connect all dropdowns to output
        submit_btn.click(
            process_selections,
            inputs=[model_dropdown, destination_dropdown, language_dropdown, priority_dropdown],
            outputs=output
        )

    return demo


# =============================================================================
# EXAMPLE 2: DEPENDENT DROPDOWNS (Dynamic Updates)
# =============================================================================

def example2_dependent_dropdowns():
    """Dropdowns that update each other based on selections"""

    def update_model_versions(provider):
        """Update model versions based on provider selection"""
        if provider in MODELS:
            return gr.Dropdown(choices=MODELS[provider], value=MODELS[provider][0])
        return gr.Dropdown(choices=[], value=None)

    def update_destination_info(destination):
        """Provide additional info based on destination"""
        info_map = {
            "London": "🇬🇧 GMT timezone, £ currency",
            "Paris": "🇫🇷 CET timezone, € currency",
            "Tokyo": "🇯🇵 JST timezone, ¥ currency",
            "Berlin": "🇩🇪 CET timezone, € currency",
            "New York": "🇺🇸 EST timezone, $ currency"
        }
        return info_map.get(destination, "No information available")

    def process_dependent_selections(provider, model, destination, language):
        result = f"""
        Final Configuration:
        - Provider: {provider}
        - Specific Model: {model}
        - Destination: {destination}
        - Language: {language}
        """
        return result

    with gr.Blocks(title="Example 2: Dependent Dropdowns") as demo:
        gr.Markdown("# Dependent Dropdowns Example")
        #gr.Markdown("*Model versions update based on provider selection*")

        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                label="Model Provider",
                value="OpenAI"
            )
            # This dropdown updates based on provider selection
            model_dropdown = gr.Dropdown(
                choices=MODELS["OpenAI"],  # Default to OpenAI models
                label="Specific Model",
                value=MODELS["OpenAI"][0]
            )

        with gr.Row():
            destination_dropdown = gr.Dropdown(
                choices=DESTINATIONS,
                label="Destination",
                value="London"
            )
            language_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                label="Response Language",
                value="English"
            )

        # Info box that updates with destination
        destination_info = gr.Textbox(
            label="Destination Info",
            value="🇬🇧 GMT timezone, £ currency",
            interactive=False
        )

        output = gr.Textbox(label="Final Configuration", lines=5)
        submit_btn = gr.Button("Apply Configuration")

        # Set up dependencies
        provider_dropdown.change(
            update_model_versions,
            inputs=provider_dropdown,
            outputs=model_dropdown
        )

        destination_dropdown.change(
            update_destination_info,
            inputs=destination_dropdown,
            outputs=destination_info
        )

        submit_btn.click(
            process_dependent_selections,
            inputs=[provider_dropdown, model_dropdown, destination_dropdown, language_dropdown],
            outputs=output
        )

    return demo


# =============================================================================
# EXAMPLE 3: FLIGHTAI INTEGRATION (Your Use Case)
# =============================================================================

def example3_flightai_integration():
    """Integration with your FlightAI chatbot logic"""

    # Mock your existing data structures
    ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}
    available_models = {"OpenAI": "gpt-4o-mini", "Anthropic": "claude-3-sonnet", "Google": "gemini-pro"}

    def get_price_and_model_info(destination, provider, language, priority):
        """Simulate your existing functions with multiple dropdown inputs"""

        # Get price (like your get_ticket_price function)
        price = ticket_prices.get(destination.lower(), "Unknown")

        # Get model (like your get_model function)
        model = available_models.get(provider, "gpt-4o-mini")

        # Create response based on all selections
        response = f"""
        ✈️ FlightAI Configuration:

        🎯 Destination: {destination}
        💰 Price: {price}
        🤖 AI Model: {model} ({provider})
        🌍 Language: {language}
        🎫 Service Level: {priority}

        Ready to assist you in {language} for your {priority} class trip to {destination}!
        """

        return response

    def simulate_chat_with_dropdowns(message, history, destination, provider, language, priority):
        """Simulate your chat function with multiple dropdown inputs"""

        if not message.strip():
            return history

        # Add user message
        history = history + [{"role": "user", "content": message}]

        # Create context-aware response based on all dropdown selections
        model = available_models.get(provider, "gpt-4o-mini")
        price = ticket_prices.get(destination.lower(), "Unknown")

        ai_response = f"[Using {model}] Hello! I can help you with your {priority} class trip to {destination} (price: {price}). How can I assist you today?"

        # Add AI response
        history = history + [{"role": "assistant", "content": ai_response}]

        return history

    with gr.Blocks(title="Example 3: FlightAI Integration") as demo:
        gr.Markdown("# FlightAI with Multiple Dropdowns")

        # Configuration Panel
        with gr.Accordion("Flight & AI Configuration", open=True):
            with gr.Row():
                destination_dropdown = gr.Dropdown(
                    choices=DESTINATIONS,
                    label="🎯 Destination",
                    value="London"
                )
                provider_dropdown = gr.Dropdown(
                    choices=list(available_models.keys()),
                    label="🤖 AI Provider",
                    value="OpenAI"
                )

            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=LANGUAGES,
                    label="🌍 Response Language",
                    value="English"
                )
                priority_dropdown = gr.Dropdown(
                    choices=PRIORITIES,
                    label="🎫 Service Level",
                    value="Economy"
                )

        # Configuration Display
        config_output = gr.Textbox(
            label="Current Configuration",
            lines=8,
            value="Select your preferences above"
        )

        update_config_btn = gr.Button("🔄 Update Configuration")

        # Chat Interface
        gr.Markdown("---")
        gr.Markdown("## Chat Interface")

        chatbot = gr.Chatbot(height=300, type="messages")

        with gr.Row():
            chat_input = gr.Textbox(
                label="Message",
                placeholder="Ask about flights, prices, or anything else...",
                scale=4
            )
            send_btn = gr.Button("Send", scale=1)

        clear_btn = gr.Button("Clear Chat")

        # Connect configuration update
        update_config_btn.click(
            get_price_and_model_info,
            inputs=[destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown],
            outputs=config_output
        )

        # Connect chat functionality
        def handle_chat_submit(message, history, dest, prov, lang, prio):
            return "", simulate_chat_with_dropdowns(message, history, dest, prov, lang, prio)

        send_btn.click(
            handle_chat_submit,
            inputs=[chat_input, chatbot, destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown],
            outputs=[chat_input, chatbot]
        )

        chat_input.submit(
            handle_chat_submit,
            inputs=[chat_input, chatbot, destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown],
            outputs=[chat_input, chatbot]
        )

        clear_btn.click(lambda: [], outputs=chatbot)

        # Auto-update config on any dropdown change
        for dropdown in [destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown]:
            dropdown.change(
                get_price_and_model_info,
                inputs=[destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown],
                outputs=config_output
            )

    return demo


# =============================================================================
# EXAMPLE 4: ADVANCED PATTERNS
# =============================================================================

def example4_advanced_patterns():
    """Advanced dropdown patterns with validation and complex interactions"""

    def validate_combination(destination, priority, language):
        """Validate that certain combinations are allowed"""

        # Example business rules
        restrictions = {
            "Tokyo": {"forbidden_priority": "Economy", "required_language": "Japanese"},
            "Berlin": {"forbidden_language": "Japanese"}
        }

        warnings = []

        if destination in restrictions:
            rules = restrictions[destination]

            if "forbidden_priority" in rules and priority == rules["forbidden_priority"]:
                warnings.append(f"⚠️ {priority} class not available for {destination}")

            if "required_language" in rules and language != rules["required_language"]:
                warnings.append(
                    f"💡 Consider selecting {rules['required_language']} for better service in {destination}")

            if "forbidden_language" in rules and language == rules["forbidden_language"]:
                warnings.append(f"❌ {language} support limited in {destination}")

        if warnings:
            return "⚠️ Validation Issues:\n" + "\n".join(warnings)
        else:
            return "✅ Configuration looks good!"

    def get_filtered_options(selection_type, current_selections):
        """Filter dropdown options based on other selections"""
        # This simulates complex business logic for available options

        if selection_type == "priority" and current_selections.get("destination") == "Tokyo":
            # Remove Economy for Tokyo
            return [p for p in PRIORITIES if p != "Economy"]

        return PRIORITIES  # Return all if no restrictions

    def handle_complex_interaction(dest, prov, lang, prio, message):
        """Handle complex interactions between all components"""

        # Validation
        validation_result = validate_combination(dest, prio, lang)

        # Response generation based on all inputs
        if "⚠️" in validation_result or "❌" in validation_result:
            response = f"Configuration Issue Detected:\n{validation_result}\n\nPlease adjust your selections."
        else:
            response = f"""
            ✅ Perfect Configuration!

            Flight Details:
            - Destination: {dest}
            - Service Level: {prio}
            - AI Provider: {prov}
            - Communication: {lang}

            User Message: "{message}"

            I'm ready to help you plan your {prio} class trip to {dest} in {lang}!
            """

        return response

    with gr.Blocks(title="Example 4: Advanced Patterns") as demo:
        gr.Markdown("# Advanced Dropdown Patterns")
        gr.Markdown("*Includes validation, filtering, and complex business logic*")

        with gr.Row():
            destination_dropdown = gr.Dropdown(
                choices=DESTINATIONS,
                label="Destination",
                value="London"
            )
            provider_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                label="AI Provider",
                value="OpenAI"
            )

        with gr.Row():
            language_dropdown = gr.Dropdown(
                choices=LANGUAGES,
                label="Language",
                value="English"
            )
            priority_dropdown = gr.Dropdown(
                choices=PRIORITIES,
                label="Service Level",
                value="Economy"
            )

        # Validation display
        validation_output = gr.Textbox(
            label="Validation Status",
            lines=3,
            value="Make your selections above"
        )

        # Message input
        message_input = gr.Textbox(
            label="Your Message",
            placeholder="What would you like to know?",
            lines=2
        )

        # Final output
        result_output = gr.Textbox(
            label="FlightAI Response",
            lines=10
        )

        process_btn = gr.Button("🚀 Process Request")

        # Real-time validation on any change
        def update_validation(dest, prov, lang, prio):
            return validate_combination(dest, prio, lang)

        for dropdown in [destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown]:
            dropdown.change(
                update_validation,
                inputs=[destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown],
                outputs=validation_output
            )

        # Main processing
        process_btn.click(
            handle_complex_interaction,
            inputs=[destination_dropdown, provider_dropdown, language_dropdown, priority_dropdown, message_input],
            outputs=result_output
        )

    return demo


# =============================================================================
# MAIN DEMO LAUNCHER
# =============================================================================

def launch_all_examples():
    """Launch all examples in a tabbed interface"""

    with gr.Blocks(title="Multiple Dropdowns Integration Guide") as demo:
        gr.Markdown("# 🎛️ Multiple Dropdowns in Gradio - Complete Guide")
        gr.Markdown("*Choose a tab to explore different integration patterns*")

        with gr.Tabs():
            with gr.TabItem("1️⃣ Simple Independent"):
                example1_simple_dropdowns()

            with gr.TabItem("2️⃣ Dependent Updates"):
                example2_dependent_dropdowns()

            with gr.TabItem("3️⃣ FlightAI Integration"):
                example3_flightai_integration()

            with gr.TabItem("4️⃣ Advanced Patterns"):
                example4_advanced_patterns()

    return demo


# Launch the demo
if __name__ == "__main__":
    demo = launch_all_examples()
    demo.launch(inbrowser=True)