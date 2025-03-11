# Text to ASL 3D Avatar Translator
# Complete implementation using Python, Blender, and NLP techniques

import os
import json
import numpy as np
import pandas as pd
from transformers import pipeline
import bpy  # Blender Python API
from transformers import AutoTokenizer

import sys
print(sys.executable)


class TextToASLTranslator:
    def __init__(self, asl_data_path="asl_gestures_data.json"):
        """
        Initialize the Text to ASL Translator
        
        Args:
            asl_data_path: Path to the ASL gestures data
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.asl_data = self._load_asl_data(asl_data_path)
        self.setup_blender_scene()
        
    def _load_asl_data(self, data_path):
        """Load ASL gesture data from JSON file"""
        if not os.path.exists(data_path):
            print(f"Warning: ASL data file not found at {data_path}. Creating empty dictionary.")
            return {}
        
        with open(data_path, 'r') as f:
            return json.load(f)
    
    def setup_blender_scene(self):
        """Set up the Blender scene with an avatar, lighting, and a camera"""

        # Clear existing objects
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Ensure Rigify is enabled before adding the meta-rig
        if "rigify" not in bpy.context.preferences.addons:
            bpy.ops.preferences.addon_enable(module="rigify")

        # Try to add the Rigify human meta-rig
        try:
            bpy.ops.object.armature_human_meta_rig_add()
            self.avatar = bpy.context.object
        except AttributeError:
            print("Error: 'armature_human_meta_rig_add' is unavailable in this Blender version.")
            print("Attempting to add a basic armature instead...")

            # Add a basic armature as a fallback
            bpy.ops.object.armature_add()
            self.avatar = bpy.context.object
            self.avatar.name = "FallbackRig"

        # Add lighting
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))

        # Add camera
        bpy.ops.object.camera_add(location=(0, -3, 1.5), rotation=(np.radians(80), 0, 0))
        bpy.context.scene.camera = bpy.context.object

        # Optional: Add a basic plane under the avatar
        bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 0))
        bpy.ops.object.shade_smooth()

        print("Scene setup complete!")
        
    def preprocess_text(self, text):
        """
        Preprocess input text to handle grammar, convert to ASL grammar order
        
        Args:
            text: Input English text
            
        Returns:
            List of words in ASL grammar order
        """
        # Tokenize the text
        tokens = self.tokenizer(text)["input_ids"]
        
        # Convert back to words (simplified for demonstration)
        words = text.lower().split()
        
        # Apply ASL grammar rules (simplified)
        # In ASL, time indicators often come first
        time_indicators = ["yesterday", "today", "tomorrow", "now", "later"]
        question_words = ["who", "what", "where", "when", "why", "how"]
        
        # Reorder to typical ASL grammar (time, subject, object, verb, etc.)
        # This is highly simplified and not comprehensive
        asl_ordered_words = []
        
        # Extract time indicators first
        time_words = [word for word in words if word in time_indicators]
        asl_ordered_words.extend(time_words)
        
        # For questions, move question words to the end (in ASL questions often end with the question word)
        is_question = any(word in question_words for word in words) or text.strip().endswith("?")
        question_word = None
        
        if is_question:
            for word in words:
                if word in question_words:
                    question_word = word
                    break
        
        # Add other words, skipping already processed time indicators and question word
        for word in words:
            if word not in time_indicators and word != question_word:
                asl_ordered_words.append(word)
                
        # For questions, add question word at the end
        if question_word:
            asl_ordered_words.append(question_word)
            
        return asl_ordered_words
        
    def animate_sign(self, word, start_frame):
        """
        Animate the avatar to perform the sign for a given word
        
        Args:
            word: The word to sign
            start_frame: The starting frame for this animation
            
        Returns:
            End frame after this animation
        """
        if word.lower() in self.asl_data:
            gesture_data = self.asl_data[word.lower()]
            duration = gesture_data.get("duration", 10)
            
            # Set keyframes for each joint movement
            for joint, keyframes in gesture_data.get("keyframes", {}).items():
                if joint in self.avatar.pose.bones:
                    bone = self.avatar.pose.bones[joint]
                    
                    for time_offset, rotation in keyframes:
                        frame = start_frame + time_offset
                        bone.rotation_euler = (np.radians(rotation[0]), 
                                              np.radians(rotation[1]), 
                                              np.radians(rotation[2]))
                        bone.keyframe_insert(data_path="rotation_euler", frame=frame)
            
            return start_frame + duration
        else:
            # If word not found, attempt to fingerspell
            return self.fingerspell_word(word, start_frame)
    
    def fingerspell_word(self, word, start_frame):
        """
        Fingerspell a word that doesn't have a sign in our database
        
        Args:
            word: The word to fingerspell
            start_frame: The starting frame
            
        Returns:
            End frame after fingerspelling
        """
        current_frame = start_frame
        frames_per_letter = 5
        
        for letter in word.lower():
            if letter in self.asl_data.get("fingerspelling", {}):
                letter_data = self.asl_data["fingerspelling"][letter]
                
                # Set hand position for this letter
                for joint, rotation in letter_data.items():
                    if joint in self.avatar.pose.bones:
                        bone = self.avatar.pose.bones[joint]
                        bone.rotation_euler = (np.radians(rotation[0]), 
                                              np.radians(rotation[1]), 
                                              np.radians(rotation[2]))
                        bone.keyframe_insert(data_path="rotation_euler", frame=current_frame)
            
            current_frame += frames_per_letter
            
        return current_frame
    
    def translate_text_to_asl(self, text):
        """
        Translate English text to ASL animation
        
        Args:
            text: English text to translate
        """
        asl_words = self.preprocess_text(text)
        
        # Reset the animation timeline
        bpy.context.scene.frame_start = 1
        bpy.context.scene.frame_end = 1
        
        # Animate each word
        current_frame = 1
        for word in asl_words:
            current_frame = self.animate_sign(word, current_frame)
        
        # Set the end frame
        bpy.context.scene.frame_end = current_frame
        
        # Return to frame 1
        bpy.context.scene.frame_current = 1
        
    # def render_animation(self, output_path="C:\\Aura\\backend\\3D\\output_animation.mp4"):
    #     """Render the animation to a video file with proper file handling"""

    #     # Ensure the output path includes a directory
    #     if not os.path.isabs(output_path):
    #         print("Error: Output path must be an absolute path.")
    #         return

    #     output_dir = os.path.dirname(output_path)
        
    #     # Validate the output directory
    #     if not output_dir:
    #         print("Error: Output directory path is invalid.")
    #         return

    #     # Create the directory if it doesn't exist
    #     if not os.path.exists(output_dir):
    #         try:
    #             os.makedirs(output_dir)
    #             print(f"Created output directory: {output_dir}")
    #         except Exception as e:
    #             print(f"Error: Could not create output directory: {e}")
    #             return

    #     # Ensure Blender has write permissions
    #     if not os.access(output_dir, os.W_OK):
    #         print(f"Error: Blender does not have permission to write to {output_dir}")
    #         return

    #     # Set render settings
    #     scene = bpy.context.scene
    #     scene.render.filepath = output_path
    #     scene.render.image_settings.file_format = 'FFMPEG'
    #     scene.render.ffmpeg.format = 'MPEG4'
    #     scene.render.ffmpeg.codec = 'H264'  # Better compatibility
    #     scene.render.resolution_x = 1280
    #     scene.render.resolution_y = 720
    #     scene.render.ffmpeg.audio_codec = 'AAC'  # Ensures audio is included

    #     # Render animation
    #     try:
    #         bpy.ops.render.render(animation=True)
    #         print(f"Rendering completed successfully! File saved at: {output_path}")
    #     except RuntimeError as e:
    #         print(f"Rendering failed: {e}")
        

    def render_animation(self, output_path="C:\\Aura\\backend\\3D\\output_animation.mp4"):
        """Render the animation to a video file with proper file handling"""

        # Ensure the output path includes a directory
        if not os.path.isabs(output_path):
            print("Error: Output path must be an absolute path.")
            return

        output_dir = os.path.dirname(output_path)
        
        # Validate the output directory
        if not output_dir:
            print("Error: Output directory path is invalid.")
            return

        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Error: Could not create output directory: {e}")
                return

        # Ensure Blender has write permissions
        if not os.access(output_dir, os.W_OK):
            print(f"Error: Blender does not have permission to write to {output_dir}")
            return

        # Set render settings
        scene = bpy.context.scene
        scene.render.filepath = output_path
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'  # Better compatibility
        scene.render.resolution_x = 1920  # Adjust resolution for better clarity
        scene.render.resolution_y = 1080  # Adjust resolution for better clarity
        scene.render.ffmpeg.audio_codec = 'AAC'  # Ensures audio is included

        # Debugging: Print the current scene settings
        print(f"Rendering animation from frame {scene.frame_start} to {scene.frame_end}")
        
        # Render animation
        try:
            bpy.ops.render.render(animation=True)
            print(f"Rendering completed successfully! File saved at: {output_path}")
        except RuntimeError as e:
            print(f"Rendering failed: {e}")



    def create_sample_asl_data(self, output_path="asl_gestures_data.json"):
        """Create a sample ASL data file with a few gestures"""
        sample_data = {
            "hello": {
                "duration": 15,
                "keyframes": {
                    "hand_r": [
                        [0, [0, 0, 0]],
                        [5, [45, 0, 0]],
                        [10, [90, 0, 0]],
                        [15, [0, 0, 0]]
                    ],
                    "forearm_r": [
                        [0, [0, 0, 0]],
                        [7, [0, 45, 0]],
                        [15, [0, 0, 0]]
                    ]
                }
            },
            "thank you": {
                "duration": 10,
                "keyframes": {
                    "hand_r": [
                        [0, [0, 0, 0]],
                        [5, [0, 0, 45]],
                        [10, [0, 0, 0]]
                    ],
                    "forearm_r": [
                        [0, [0, 0, 0]],
                        [5, [45, 0, 0]],
                        [10, [0, 0, 0]]
                    ]
                }
            },
            "fingerspelling": {
                "a": {"thumb_r": [10, 0, 0], "index_r": [90, 0, 0]},
                "b": {"thumb_r": [0, 0, 0], "index_r": [0, 0, 0]},
                # Add more letters as needed
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"Sample ASL data file created at {output_path}")
        return sample_data

def main():

    
    # Create the translator
    translator = TextToASLTranslator()
    
    # Create sample data if real data not available
    #translator.asl_data = translator.create_sample_asl_data()
    
    # Example usage
    text = "Hello, how are you today?"
    translator.translate_text_to_asl(text)
    translator.render_animation()
    
    print(f"Animation for '{text}' has been rendered.")

if __name__ == "__main__":
    
    # Note: This script is designed to be run within Blender's Python environment
    main()