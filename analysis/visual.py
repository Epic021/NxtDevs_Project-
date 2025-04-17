import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional
import plotly.io as pio

class InterviewVisualizer:
    def __init__(self, data_dir: str = "interview_data"):
        self.data_dir = data_dir
        self.set_style()
    
    def set_style(self):
        """Set consistent style for all visualizations."""
        # Set matplotlib style
        plt.style.use('seaborn-v0_8')
        
        # Set plotly template
        pio.templates.default = "plotly_white"
        
        # Set color palette
        self.colors = {
            'primary': '#2E86C1',
            'secondary': '#28B463',
            'accent': '#F39C12',
            'background': '#FDFEFE',
            'text': '#2C3E50'
        }
    
    def load_interview_data(self, interview_number: int) -> Dict:
        """Load interview data from JSON file."""
        try:
            filename = os.path.join(self.data_dir, f"interview_{interview_number}.json")
            with open(filename, 'r') as f:
                data = json.load(f)
                
            # Ensure metrics exist
            if 'metrics' not in data:
                data['metrics'] = []
                
            # Ensure each metric has required fields
            for metric in data['metrics']:
                if 'facial' not in metric:
                    metric['facial'] = {'confidence': 0.0}
                if 'eye' not in metric:
                    metric['eye'] = {'confidence': 0.0}
                if 'speech' not in metric:
                    metric['speech'] = {
                        'confidence_score': 0.0,
                        'wpm': 0.0,
                        'filler_count': 0,
                        'pause_count': 0,
                        'avg_pause_duration': 0.0,
                        'clarity_score': 0.0,
                        'complexity_score': 0.0
                    }
                    
            return data
        except FileNotFoundError:
            print(f"Interview data file not found: {filename}")
            return {'metrics': []}
        except Exception as e:
            print(f"Error loading interview data: {str(e)}")
            return {'metrics': []}
    
    def create_spider_chart(self, data: Dict) -> go.Figure:
        """Create a simple spider chart for key metrics."""
        try:
            metrics = data.get('metrics', [])
            if not metrics:
                return self._create_empty_plot("No data available")
            
            # Get the most recent metrics for simplicity
            latest_metrics = metrics[-1]
            
            # Get key metrics
            values = [
                latest_metrics.get('overall_confidence', 0.0),
                latest_metrics.get('engagement_score', 0.0),
                latest_metrics['facial_expression'].get('confidence', 0.0),
                latest_metrics['eye_tracking'].get('confidence', 0.0),
                latest_metrics['speech_analysis'].get('confidence', 0.0)
            ]
            
            labels = ['Overall', 'Engagement', 'Face', 'Eye', 'Speech']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=labels,
                fill='toself',
                name='Latest Performance',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickformat='.0%'
                    )
                ),
                showlegend=False,
                title='Current Performance Overview',
                height=400
            )
            
            return fig
        except Exception as e:
            print(f"Error creating spider chart: {str(e)}")
            return self._create_empty_plot("Error creating visualization")
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
    
    def create_timeline_plot(self, metrics_list: List[Dict], save_path: Optional[str] = None) -> go.Figure:
        """Create a simple timeline plot."""
        try:
            # Get last 10 data points for clarity
            metrics_list = metrics_list[-10:]
            
            timestamps = [m['timestamp'].split('T')[1][:8] for m in metrics_list]  # Just show time HH:MM:SS
            confidence = [m['overall_confidence'] for m in metrics_list]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=confidence,
                name='Confidence',
                line=dict(color='blue', width=2),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title="Confidence Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence Score",
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                height=400
            )
            
            return fig
        except Exception as e:
            print(f"Error creating timeline plot: {str(e)}")
            return self._create_empty_plot("Error creating visualization")
    
    def create_emotion_heatmap(self, metrics_list: List[Dict], save_path: Optional[str] = None) -> go.Figure:
        """Create a simple bar chart of emotions instead of heatmap."""
        try:
            # Get emotions from facial expressions
            emotions = [m['facial_expression']['emotion'] for m in metrics_list]
            
            # Count occurrences of each emotion
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(emotion_counts.keys()),
                    y=list(emotion_counts.values()),
                    marker_color='blue'
                )
            ])
            
            fig.update_layout(
                title="Facial Expression Summary",
                xaxis_title="Expression",
                yaxis_title="Count",
                height=400
            )
            
            return fig
        except Exception as e:
            print(f"Error creating emotion chart: {str(e)}")
            return self._create_empty_plot("Error creating visualization")
    
    def create_speech_metrics_plot(self, metrics: Dict, save_path: Optional[str] = None) -> go.Figure:
        """Create a simple bar chart for speech metrics."""
        try:
            # Get speech metrics
            speech_data = metrics['speech_analysis']
            
            metrics_to_show = {
                'Words/Min': speech_data.get('wpm', 0),
                'Fillers': speech_data.get('filler_count', 0)
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics_to_show.keys()),
                    y=list(metrics_to_show.values()),
                    marker_color='blue'
                )
            ])
            
            fig.update_layout(
                title="Speech Metrics",
                xaxis_title="Metric",
                yaxis_title="Value",
                height=400
            )
            
            return fig
        except Exception as e:
            print(f"Error creating speech plot: {str(e)}")
            return self._create_empty_plot("Error creating visualization")
    
    def create_comprehensive_dashboard(self, interview_number: int, save_dir: Optional[str] = None) -> go.Figure:
        """Create a simple dashboard layout."""
        try:
            data = self.load_interview_data(interview_number)
            metrics_list = data.get('metrics', [])
            
            if not metrics_list:
                return self._create_empty_plot("No data available")
            
            # Create subplots with correct specifications
            fig = make_subplots(
                rows=2, cols=2,
                specs=[
                    [{"type": "polar"}, {"type": "xy"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ],
                subplot_titles=(
                    "Performance Overview",
                    "Confidence Timeline",
                    "Facial Expressions",
                    "Speech Metrics"
                )
            )
            
            # Add spider chart
            spider = self.create_spider_chart(data)
            for trace in spider.data:
                fig.add_trace(trace, row=1, col=1)
            
            # Add timeline
            timeline = self.create_timeline_plot(metrics_list)
            for trace in timeline.data:
                fig.add_trace(trace, row=1, col=2)
            
            # Add emotion chart
            emotions = self.create_emotion_heatmap(metrics_list)
            for trace in emotions.data:
                fig.add_trace(trace, row=2, col=1)
            
            # Add speech metrics
            speech = self.create_speech_metrics_plot(metrics_list[-1])
            for trace in speech.data:
                fig.add_trace(trace, row=2, col=2)
            
            # Update layout
            fig.update_layout(
                height=800,
                width=1000,
                title_text=f"Interview {interview_number} - Summary",
                showlegend=False
            )
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                fig.write_html(os.path.join(save_dir, f"interview_{interview_number}_dashboard.html"))
            
            return fig
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            return self._create_empty_plot("Error creating dashboard")

    def create_facial_analysis_plot(self, metrics_list: List[Dict]) -> go.Figure:
        """Create detailed facial analysis visualization."""
        try:
            # Get facial metrics
            emotions = [m['facial_expression']['emotion'] for m in metrics_list]
            confidences = [m['facial_expression']['confidence'] for m in metrics_list]
            timestamps = [m['timestamp'].split('T')[1][:8] for m in metrics_list]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Emotion Distribution", "Confidence Over Time")
            )
            
            # Emotion distribution
            emotion_counts = {}
            for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            fig.add_trace(
                go.Bar(
                    x=list(emotion_counts.keys()),
                    y=list(emotion_counts.values()),
                    marker_color='blue',
                    name='Emotions'
                ),
                row=1, col=1
            )
            
            # Confidence timeline
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    name='Confidence'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=800,
                title='Facial Analysis',
                showlegend=False
            )
            
            return fig
        except Exception as e:
            print(f"Error in facial analysis: {str(e)}")
            return self._create_empty_plot("Error")

    def create_eye_analysis_plot(self, metrics_list: List[Dict]) -> go.Figure:
        """Create detailed eye tracking visualization."""
        try:
            # Get eye metrics
            confidences = [m['eye_tracking']['confidence'] for m in metrics_list]
            timestamps = [m['timestamp'].split('T')[1][:8] for m in metrics_list]
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Confidence Over Time", "Confidence Distribution")
            )
            
            # Confidence timeline
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    name='Confidence'
                ),
                row=1, col=1
            )
            
            # Confidence histogram
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    marker_color='blue',
                    name='Distribution'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=800,
                title='Eye Tracking Analysis',
                showlegend=False
            )
            
            return fig
        except Exception as e:
            print(f"Error in eye analysis: {str(e)}")
            return self._create_empty_plot("Error")

    def create_speech_analysis_plot(self, metrics_list: List[Dict]) -> go.Figure:
        """Create detailed speech analysis visualization."""
        try:
            # Get speech metrics
            wpm = [m['speech_analysis'].get('wpm', 0) for m in metrics_list]
            fillers = [m['speech_analysis'].get('filler_count', 0) for m in metrics_list]
            confidences = [m['speech_analysis'].get('confidence', 0) for m in metrics_list]
            timestamps = [m['timestamp'].split('T')[1][:8] for m in metrics_list]
            
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Words per Minute", "Filler Words", "Confidence")
            )
            
            # WPM timeline
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=wpm,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    name='WPM'
                ),
                row=1, col=1
            )
            
            # Filler words timeline
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=fillers,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    name='Fillers'
                ),
                row=2, col=1
            )
            
            # Confidence timeline
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode='lines+markers',
                    line=dict(color='blue'),
                    name='Confidence'
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=1000,
                title='Speech Analysis',
                showlegend=False
            )
            
            return fig
        except Exception as e:
            print(f"Error in speech analysis: {str(e)}")
            return self._create_empty_plot("Error")

    def save_visualizations(self, interview_number: int, save_dir: str = "visualizations"):
        """Save all visualizations to HTML files."""
        try:
            data = self.load_interview_data(interview_number)
            metrics_list = data.get('metrics', [])
            
            if not metrics_list:
                print("No data available")
                return
            
            os.makedirs(save_dir, exist_ok=True)
            
            # Create and save each analysis plot
            facial = self.create_facial_analysis_plot(metrics_list)
            facial.write_html(os.path.join(save_dir, f"interview_{interview_number}_facial.html"))
            
            eye = self.create_eye_analysis_plot(metrics_list)
            eye.write_html(os.path.join(save_dir, f"interview_{interview_number}_eye.html"))
            
            speech = self.create_speech_analysis_plot(metrics_list)
            speech.write_html(os.path.join(save_dir, f"interview_{interview_number}_speech.html"))
            
            print(f"Visualizations saved to {save_dir}")
            
        except Exception as e:
            print(f"Error saving visualizations: {str(e)}")

def visualize_interview(interview_number: int, save_dir: str = "visualizations"):
    """Main function to create visualizations."""
    visualizer = InterviewVisualizer()
    visualizer.save_visualizations(interview_number, save_dir)

if __name__ == "__main__":
    # Example usage
    visualize_interview(1)
