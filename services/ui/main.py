"""
UI Service
Gradio interface for PPG glucose monitoring
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import numpy as np
import logging
from typing import Optional
from datetime import datetime
import asyncio
from threading import Lock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UI Service")

class UpdateResultRequest(BaseModel):
    glucose: float
    num_segments: int
    quality_score: float
    device: str

# Global state for results
class ResultState:
    def __init__(self):
        self.lock = Lock()
        self.latest_result = None
        self.history = []
        
    def update(self, glucose: float, num_segments: int, quality_score: float, device: str):
        with self.lock:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result = {
                "timestamp": timestamp,
                "glucose": glucose,
                "num_segments": num_segments,
                "quality_score": quality_score,
                "device": device
            }
            self.latest_result = result
            self.history.append(result)
            # Keep last 50 results
            if len(self.history) > 50:
                self.history = self.history[-50:]
            logger.info(f"Updated result: {glucose:.1f} mg/dL")
    
    def get_latest(self):
        with self.lock:
            return self.latest_result
    
    def get_history(self):
        with self.lock:
            return self.history.copy()

result_state = ResultState()

@app.post("/update_result")
async def update_result(request: UpdateResultRequest):
    """Receive prediction result from model service"""
    try:
        result_state.update(
            request.glucose,
            request.num_segments,
            request.quality_score,
            request.device
        )
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to update result: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/health")
async def health():
    return {"service": "UI Service", "status": "ready"}

# Gradio Interface
def create_gradio_interface():
    """Create Gradio UI"""
    
    def get_current_result():
        """Get latest prediction result"""
        result = result_state.get_latest()
        
        if result is None:
            return (
                "No prediction yet",
                "Waiting for data...",
                "",
                "",
                ""
            )
        
        glucose = result["glucose"]
        
        # Determine glucose level status
        if glucose < 70:
            status = "⚠️ LOW"
            status_color = "🔴"
        elif glucose <= 140:
            status = "✅ NORMAL"
            status_color = "🟢"
        elif glucose <= 200:
            status = "⚠️ ELEVATED"
            status_color = "🟡"
        else:
            status = "⚠️ HIGH"
            status_color = "🔴"
        
        glucose_display = f"{status_color} {glucose:.1f} mg/dL - {status}"
        
        details = f"""
**Timestamp:** {result['timestamp']}
**Quality Score:** {result['quality_score']:.2%}
**Segments Analyzed:** {result['num_segments']}
**Device:** {result['device']}
        """
        
        # Clinical ranges reference
        ranges = """
### Reference Ranges:
- **< 70 mg/dL:** Hypoglycemia (Low) 🔴
- **70-140 mg/dL:** Normal 🟢
- **141-200 mg/dL:** Elevated 🟡
- **> 200 mg/dL:** Hyperglycemia (High) 🔴
        """
        
        return (
            glucose_display,
            details,
            ranges,
            _get_history_text(),
            _get_history_plot()
        )
    
    def _get_history_text():
        """Get history as text"""
        history = result_state.get_history()
        if not history:
            return "No history yet"
        
        lines = ["### Recent Predictions:\n"]
        for r in reversed(history[-10:]):  # Last 10
            lines.append(
                f"- **{r['timestamp']}:** {r['glucose']:.1f} mg/dL "
                f"(Quality: {r['quality_score']:.1%}, Segments: {r['num_segments']})"
            )
        
        return "\n".join(lines)
    
    def _get_history_plot():
        """Generate history plot"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            
            history = result_state.get_history()
            if len(history) < 2:
                return None
            
            times = [r['timestamp'] for r in history]
            glucose_values = [r['glucose'] for r in history]
            
            fig = Figure(figsize=(10, 5))
            ax = fig.add_subplot(111)
            
            ax.plot(range(len(glucose_values)), glucose_values, 'bo-', linewidth=2, markersize=8)
            ax.axhline(y=70, color='r', linestyle='--', label='Low threshold')
            ax.axhline(y=140, color='g', linestyle='--', label='Normal upper')
            ax.axhline(y=200, color='orange', linestyle='--', label='High threshold')
            
            ax.set_xlabel('Measurement Number')
            ax.set_ylabel('Glucose (mg/dL)')
            ax.set_title('Glucose Level History')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return fig
            
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")
            return None
    
    # Build Gradio interface
    with gr.Blocks(title="PPG Glucose Monitor", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🩸 PPG-Based Glucose Monitor
            
            Real-time glucose prediction from PPG (Photoplethysmography) signals.
            
            This system uses a deep learning model (TSEncoder) to predict blood glucose levels 
            from 1-minute PPG recordings collected via ESP32 + MAX30102 sensor.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main result display
                glucose_output = gr.Markdown(
                    value="## Waiting for data...",
                    elem_id="glucose-display"
                )
                
                details_output = gr.Markdown(
                    value="No prediction yet"
                )
                
                # Auto-refresh button
                refresh_btn = gr.Button("🔄 Refresh", variant="primary")
        
            with gr.Column(scale=1):
                ranges_output = gr.Markdown(
                    value="""
### Reference Ranges:
- **< 70 mg/dL:** Hypoglycemia (Low) 🔴
- **70-140 mg/dL:** Normal 🟢
- **141-200 mg/dL:** Elevated 🟡
- **> 200 mg/dL:** Hyperglycemia (High) 🔴
                    """
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 📊 History")
                history_text = gr.Markdown(value="No history yet")
        
        with gr.Row():
            with gr.Column():
                history_plot = gr.Plot(label="Glucose Trends")
        
        # Manual refresh button
        refresh_btn.click(
            fn=get_current_result,
            inputs=[],
            outputs=[glucose_output, details_output, ranges_output, history_text, history_plot]
        )

        # Initial load
        interface.load(
            fn=get_current_result,
            inputs=[],
            outputs=[glucose_output, details_output, ranges_output, history_text, history_plot]
        )

        # Auto-refresh every 3 seconds
        interface.load(
            fn=get_current_result,
            inputs=[],
            outputs=[glucose_output, details_output, ranges_output, history_text, history_plot],
            every=3
        )
    
    return interface

# Mount Gradio app
gradio_app = create_gradio_interface()

# Mount Gradio to FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting UI Service with Gradio interface")
    uvicorn.run(app, host="0.0.0.0", port=8003)
