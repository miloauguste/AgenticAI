#!/usr/bin/env python3
"""
Demo script showing how to see AgentState at any given time
This demonstrates multiple ways to inspect and monitor state
"""

import json
import uuid
from datetime import datetime, timezone
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import time

# AgentState definition
class AgentState(TypedDict):
    messages: List[Any]
    user_id: str
    thread_id: str
    user_profile: Dict[str, Any]
    query_history: List[Dict[str, Any]]
    current_query_type: Optional[str]
    escalation_required: bool
    session_metadata: Dict[str, Any]

class StateVisibilityDemo:
    """Demonstrates various ways to see AgentState values"""
    
    def __init__(self):
        self.state_snapshots = []
        self.node_executions = []
    
    def print_state_summary(self, state: AgentState, title: str = "Current State"):
        """Method 1: Print a summary of the current state"""
        
        print(f"\n{'='*60}")
        print(f"📊 {title.upper()}")
        print(f"{'='*60}")
        
        # Basic metrics
        print(f"🆔 User ID: {state['user_id'][:8]}...")
        print(f"🧵 Thread ID: {state['thread_id'][:8]}...")
        print(f"💬 Messages: {len(state['messages'])}")
        print(f"🏷️  Query Type: {state.get('current_query_type', 'Not set')}")
        print(f"🚨 Escalation Required: {state.get('escalation_required', False)}")
        
        # Message breakdown
        if state['messages']:
            human_msgs = sum(1 for msg in state['messages'] if isinstance(msg, HumanMessage))
            ai_msgs = sum(1 for msg in state['messages'] if isinstance(msg, AIMessage))
            system_msgs = sum(1 for msg in state['messages'] if isinstance(msg, SystemMessage))
            
            print(f"   👤 Human: {human_msgs}, 🤖 AI: {ai_msgs}, ⚙️ System: {system_msgs}")
        
        # User profile info
        profile = state.get('user_profile', {})
        if profile:
            print(f"👤 Profile: {profile.get('name', 'Anonymous')} ({profile.get('subscription_type', 'Basic')})")
        
        # Metadata highlights
        metadata = state.get('session_metadata', {})
        if metadata:
            print(f"📈 Metadata Keys: {list(metadata.keys())}")
        
        print(f"{'='*60}\n")
    
    def print_full_state_json(self, state: AgentState, title: str = "Full State JSON"):
        """Method 2: Print complete state as formatted JSON"""
        
        print(f"\n{'='*60}")
        print(f"📋 {title.upper()}")
        print(f"{'='*60}")
        
        # Convert to JSON-serializable format
        json_state = self._serialize_state_for_json(state)
        
        # Pretty print JSON
        print(json.dumps(json_state, indent=2, default=str))
        print(f"{'='*60}\n")
    
    def _serialize_state_for_json(self, state: AgentState) -> Dict[str, Any]:
        """Convert state to JSON-serializable format"""
        
        serialized = {}
        
        for key, value in state.items():
            if key == "messages":
                serialized[key] = [
                    {
                        "type": type(msg).__name__,
                        "content": msg.content if hasattr(msg, 'content') else str(msg),
                        "content_length": len(msg.content) if hasattr(msg, 'content') else 0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    for msg in value
                ]
            elif isinstance(value, dict):
                serialized[key] = value
            elif isinstance(value, list):
                serialized[key] = [str(item) for item in value]
            else:
                serialized[key] = str(value)
        
        return serialized
    
    def inspect_state_detailed(self, state: AgentState, title: str = "Detailed State Inspection"):
        """Method 3: Detailed inspection with analysis"""
        
        print(f"\n{'='*80}")
        print(f"🔍 {title.upper()}")
        print(f"{'='*80}")
        
        # 1. State Overview
        print("📊 STATE OVERVIEW:")
        print(f"   Total fields: {len(state)}")
        print(f"   State size (approx): {len(str(state)):,} characters")
        
        # 2. Field-by-field analysis
        print("\n📋 FIELD ANALYSIS:")
        
        for field_name, field_value in state.items():
            print(f"\n   🔸 {field_name}:")
            print(f"      Type: {type(field_value).__name__}")
            
            if isinstance(field_value, list):
                print(f"      Length: {len(field_value)}")
                if field_name == "messages" and field_value:
                    print(f"      Message types: {[type(msg).__name__ for msg in field_value]}")
                    
                    # Show last few messages
                    print(f"      Recent messages:")
                    for i, msg in enumerate(field_value[-3:]):  # Last 3 messages
                        msg_preview = msg.content[:50] + "..." if hasattr(msg, 'content') and len(msg.content) > 50 else getattr(msg, 'content', str(msg))
                        print(f"        [{len(field_value)-3+i}] {type(msg).__name__}: {msg_preview}")
            
            elif isinstance(field_value, dict):
                print(f"      Keys: {list(field_value.keys())}")
                if field_value:
                    print(f"      Content preview: {str(field_value)[:100]}...")
            
            elif isinstance(field_value, str):
                print(f"      Length: {len(field_value)}")
                print(f"      Preview: {field_value[:50]}..." if len(field_value) > 50 else f"      Value: {field_value}")
            
            else:
                print(f"      Value: {field_value}")
        
        # 3. State Health Check
        print("\n🏥 STATE HEALTH CHECK:")
        health_issues = []
        
        # Check required fields
        required_fields = ["messages", "user_id", "thread_id"]
        for field in required_fields:
            if field not in state:
                health_issues.append(f"Missing required field: {field}")
        
        # Check message count
        if len(state.get("messages", [])) > 20:
            health_issues.append("High message count - consider trimming")
        
        # Check state size
        state_size = len(str(state))
        if state_size > 50000:  # 50KB
            health_issues.append("Large state size - performance impact possible")
        
        if health_issues:
            for issue in health_issues:
                print(f"   ⚠️  {issue}")
        else:
            print("   ✅ State appears healthy")
        
        print(f"{'='*80}\n")
    
    def create_state_snapshot(self, state: AgentState, context: str = ""):
        """Method 4: Create timestamped snapshots for comparison"""
        
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "state": self._deep_copy_state(state),
            "metrics": {
                "message_count": len(state.get("messages", [])),
                "state_size": len(str(state)),
                "field_count": len(state),
                "has_user_profile": bool(state.get("user_profile")),
                "escalation_required": state.get("escalation_required", False)
            }
        }
        
        self.state_snapshots.append(snapshot)
        
        print(f"📸 State snapshot created: {context} ({snapshot['timestamp']})")
        return len(self.state_snapshots) - 1  # Return snapshot index
    
    def compare_snapshots(self, snapshot1_idx: int, snapshot2_idx: int):
        """Method 5: Compare two state snapshots"""
        
        if snapshot1_idx >= len(self.state_snapshots) or snapshot2_idx >= len(self.state_snapshots):
            print("❌ Invalid snapshot indices")
            return
        
        snap1 = self.state_snapshots[snapshot1_idx]
        snap2 = self.state_snapshots[snapshot2_idx]
        
        print(f"\n{'='*80}")
        print(f"🔄 SNAPSHOT COMPARISON")
        print(f"{'='*80}")
        
        print(f"📸 Snapshot 1: {snap1['context']} ({snap1['timestamp']})")
        print(f"📸 Snapshot 2: {snap2['context']} ({snap2['timestamp']})")
        
        # Compare metrics
        print(f"\n📊 METRIC CHANGES:")
        for metric, value1 in snap1['metrics'].items():
            value2 = snap2['metrics'].get(metric, "N/A")
            
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                change = value2 - value1
                change_str = f"{change:+}" if change != 0 else "no change"
                print(f"   {metric}: {value1} → {value2} ({change_str})")
            else:
                print(f"   {metric}: {value1} → {value2}")
        
        # Field-level comparison
        print(f"\n🔍 FIELD CHANGES:")
        state1 = snap1['state']
        state2 = snap2['state']
        
        all_fields = set(state1.keys()) | set(state2.keys())
        
        for field in sorted(all_fields):
            if field not in state1:
                print(f"   + {field}: Added")
            elif field not in state2:
                print(f"   - {field}: Removed")
            elif str(state1[field]) != str(state2[field]):
                print(f"   ~ {field}: Changed")
                
                if field == "messages":
                    count1 = len(state1[field]) if isinstance(state1[field], list) else 0
                    count2 = len(state2[field]) if isinstance(state2[field], list) else 0
                    print(f"     Messages: {count1} → {count2}")
            else:
                print(f"   = {field}: Unchanged")
        
        print(f"{'='*80}\n")
    
    def _deep_copy_state(self, state: AgentState) -> Dict[str, Any]:
        """Create a deep copy of state for snapshots"""
        
        copied_state = {}
        
        for key, value in state.items():
            if key == "messages":
                copied_state[key] = [
                    {
                        "type": type(msg).__name__,
                        "content": getattr(msg, 'content', str(msg))
                    }
                    for msg in value
                ]
            elif isinstance(value, dict):
                copied_state[key] = value.copy()
            elif isinstance(value, list):
                copied_state[key] = value.copy()
            else:
                copied_state[key] = value
        
        return copied_state
    
    def monitor_node_execution(self, node_name: str):
        """Method 6: Decorator to monitor node execution and state changes"""
        
        def decorator(node_func):
            def wrapper(state: AgentState) -> AgentState:
                
                # Capture input state
                input_snapshot = self.create_state_snapshot(state, f"{node_name}_input")
                
                start_time = time.time()
                
                print(f"\n🚀 EXECUTING NODE: {node_name}")
                print(f"   Input state size: {len(str(state)):,} chars")
                print(f"   Input messages: {len(state.get('messages', []))}")
                
                # Execute the actual node
                try:
                    result_state = node_func(state)
                    execution_time = time.time() - start_time
                    
                    # Capture output state
                    output_snapshot = self.create_state_snapshot(result_state, f"{node_name}_output")
                    
                    print(f"   ✅ Execution completed in {execution_time:.3f}s")
                    print(f"   Output state size: {len(str(result_state)):,} chars")
                    print(f"   Output messages: {len(result_state.get('messages', []))}")
                    
                    # Log execution details
                    execution_record = {
                        "node_name": node_name,
                        "execution_time": execution_time,
                        "input_snapshot_idx": input_snapshot,
                        "output_snapshot_idx": output_snapshot,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "success": True
                    }
                    
                    self.node_executions.append(execution_record)
                    
                    # Quick comparison
                    if len(str(state)) != len(str(result_state)):
                        size_change = len(str(result_state)) - len(str(state))
                        print(f"   📏 State size change: {size_change:+} chars")
                    
                    return result_state
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    
                    print(f"   ❌ Execution failed after {execution_time:.3f}s")
                    print(f"   Error: {str(e)}")
                    
                    # Log failed execution
                    execution_record = {
                        "node_name": node_name,
                        "execution_time": execution_time,
                        "input_snapshot_idx": input_snapshot,
                        "output_snapshot_idx": None,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "success": False,
                        "error": str(e)
                    }
                    
                    self.node_executions.append(execution_record)
                    
                    # Return original state on error
                    return state
            
            return wrapper
        return decorator
    
    def print_execution_summary(self):
        """Method 7: Print summary of all node executions"""
        
        if not self.node_executions:
            print("No node executions recorded yet.")
            return
        
        print(f"\n{'='*80}")
        print(f"📈 EXECUTION SUMMARY")
        print(f"{'='*80}")
        
        total_executions = len(self.node_executions)
        successful_executions = sum(1 for exec in self.node_executions if exec['success'])
        total_time = sum(exec['execution_time'] for exec in self.node_executions)
        
        print(f"Total Executions: {total_executions}")
        print(f"Successful: {successful_executions}")
        print(f"Failed: {total_executions - successful_executions}")
        print(f"Total Execution Time: {total_time:.3f}s")
        print(f"Average Execution Time: {total_time/total_executions:.3f}s")
        
        # Node breakdown
        node_stats = {}
        for exec in self.node_executions:
            node = exec['node_name']
            if node not in node_stats:
                node_stats[node] = {'count': 0, 'total_time': 0, 'failures': 0}
            
            node_stats[node]['count'] += 1
            node_stats[node]['total_time'] += exec['execution_time']
            if not exec['success']:
                node_stats[node]['failures'] += 1
        
        print(f"\n📊 NODE STATISTICS:")
        for node, stats in node_stats.items():
            avg_time = stats['total_time'] / stats['count']
            success_rate = ((stats['count'] - stats['failures']) / stats['count']) * 100
            
            print(f"   {node}:")
            print(f"     Executions: {stats['count']}")
            print(f"     Avg Time: {avg_time:.3f}s")
            print(f"     Success Rate: {success_rate:.1f}%")
        
        print(f"{'='*80}\n")

def demo_state_visibility():
    """Main demo showing all state visibility methods"""
    
    print("🎯 LANGGRAPH STATE VISIBILITY DEMO")
    print("=" * 80)
    
    # Initialize demo
    demo = StateVisibilityDemo()
    
    # Create sample state
    sample_state: AgentState = {
        "messages": [
            SystemMessage(content="You are a helpful customer support agent."),
            HumanMessage(content="I forgot my password and can't log in to my account."),
        ],
        "user_id": str(uuid.uuid4()),
        "thread_id": str(uuid.uuid4()),
        "user_profile": {
            "name": "John Doe",
            "email": "john@example.com",
            "subscription_type": "Premium"
        },
        "query_history": [],
        "current_query_type": "authentication",
        "escalation_required": False,
        "session_metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_length": 2,
            "classification_confidence": 0.95
        }
    }
    
    print("📝 Created sample AgentState with authentication query")
    
    # Method 1: Print summary
    demo.print_state_summary(sample_state, "Initial State")
    
    # Method 2: Create snapshot
    snapshot1_idx = demo.create_state_snapshot(sample_state, "After user query")
    
    # Method 3: Simulate processing - add AI response
    sample_state["messages"].append(
        AIMessage(content="I'll help you reset your password. Please check your email for a reset link.")
    )
    sample_state["session_metadata"]["last_response_time"] = datetime.now(timezone.utc).isoformat()
    
    # Method 4: Another snapshot
    snapshot2_idx = demo.create_state_snapshot(sample_state, "After AI response")
    
    # Method 5: Detailed inspection
    demo.inspect_state_detailed(sample_state, "State After AI Response")
    
    # Method 6: Compare snapshots
    demo.compare_snapshots(snapshot1_idx, snapshot2_idx)
    
    # Method 7: Show full JSON
    demo.print_full_state_json(sample_state, "Complete State JSON")
    
    # Method 8: Demonstrate monitored node execution
    print("🔧 Demonstrating monitored node execution...")
    
    @demo.monitor_node_execution("demo_classification_node")
    def sample_classification_node(state: AgentState) -> AgentState:
        """Sample node that modifies state"""
        
        # Simulate processing time
        time.sleep(0.1)
        
        # Modify state
        if state["messages"]:
            last_msg = state["messages"][-1]
            if isinstance(last_msg, HumanMessage):
                query = last_msg.content.lower()
                if "password" in query:
                    state["current_query_type"] = "authentication"
                    state["session_metadata"]["confidence"] = 0.98
        
        return state
    
    # Execute monitored node
    sample_state = sample_classification_node(sample_state)
    
    # Print execution summary
    demo.print_execution_summary()
    
    print("✨ Demo completed! You now know how to see AgentState at any time!")
    
    return demo, sample_state

def practical_usage_examples():
    """Show practical ways to use state visibility in real applications"""
    
    print("\n" + "="*80)
    print("💡 PRACTICAL USAGE EXAMPLES")
    print("="*80)
    
    print("""
1. 🐛 DEBUGGING: Add state inspection in your nodes
   
   def my_node(state: AgentState) -> AgentState:
       # Debug: Print state before processing
       print(f"Input state: {len(state['messages'])} messages")
       
       # Your processing logic here...
       result = process_logic(state)
       
       # Debug: Print state after processing
       print(f"Output state: {len(result['messages'])} messages")
       return result

2. 📊 MONITORING: Track state changes over time
   
   state_history = []
   
   def monitored_node(state):
       # Capture state before
       before = len(json.dumps(str(state)))
       
       result = your_node_function(state)
       
       # Capture state after
       after = len(json.dumps(str(result)))
       
       state_history.append({
           'timestamp': datetime.now().isoformat(),
           'size_before': before,
           'size_after': after,
           'size_change': after - before
       })
       
       return result

3. 🔍 INSPECTION: Real-time state viewer
   
   def inspect_state_anytime(state: AgentState):
       \"\"\"Call this anywhere to see current state\"\"\"
       
       print("\\n🔍 CURRENT STATE INSPECTION:")
       print(f"Messages: {len(state.get('messages', []))}")
       print(f"User: {state.get('user_id', 'Unknown')[:8]}...")
       print(f"Query Type: {state.get('current_query_type', 'None')}")
       print(f"Escalation: {state.get('escalation_required', False)}")
       
       # Show recent messages
       messages = state.get('messages', [])
       if messages:
           print("Recent messages:")
           for msg in messages[-3:]:
               msg_type = type(msg).__name__
               content = getattr(msg, 'content', str(msg))[:50]
               print(f"  {msg_type}: {content}...")

4. ⚡ PERFORMANCE: Track execution times
   
   def performance_wrapper(node_name):
       def decorator(node_func):
           def wrapper(state):
               start = time.time()
               result = node_func(state)
               duration = time.time() - start
               
               print(f"⏱️  {node_name}: {duration:.3f}s")
               return result
           return wrapper
       return decorator

5. 💾 PERSISTENCE: Save states to file
   
   def save_state_to_file(state: AgentState, filename: str):
       \"\"\"Save state for later analysis\"\"\"
       
       serialized = {
           'timestamp': datetime.now().isoformat(),
           'user_id': state.get('user_id'),
           'message_count': len(state.get('messages', [])),
           'query_type': state.get('current_query_type'),
           'full_state': str(state)  # or use JSON serialization
       }
       
       with open(filename, 'w') as f:
           json.dump(serialized, f, indent=2)

6. 🚨 ALERTS: Set up state-based alerts
   
   def check_state_health(state: AgentState):
       \"\"\"Alert on concerning state conditions\"\"\"
       
       alerts = []
       
       # Check message count
       if len(state.get('messages', [])) > 50:
           alerts.append("High message count - memory usage concern")
       
       # Check state size
       state_size = len(str(state))
       if state_size > 100000:  # 100KB
           alerts.append("Large state size - performance impact")
       
       # Check for errors
       if 'error' in state.get('session_metadata', {}):
           alerts.append("Error detected in session metadata")
       
       for alert in alerts:
           print(f"🚨 ALERT: {alert}")
       
       return len(alerts) == 0  # Return True if healthy

""")
    
    print("="*80)
    print("🎯 KEY TAKEAWAYS:")
    print("1. You can inspect AgentState at ANY point in your workflow")
    print("2. Use state snapshots to compare before/after changes")
    print("3. Monitor performance by tracking state size and execution time")
    print("4. Debug issues by printing state details at problematic nodes")
    print("5. Set up health checks to catch problems early")
    print("="*80)

if __name__ == "__main__":
    # Run the demo
    demo_instance, final_state = demo_state_visibility()
    
    # Show practical examples
    practical_usage_examples()
    
    print("\n🎉 You now have complete visibility into your AgentState!")
    print("Try integrating these techniques into your customer support agent!")