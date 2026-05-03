import re
import os
import json
from typing import List, Dict
import argparse

try:
    from docx import Document
except ImportError:
    Document = None

try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

class SceneParser:
    def __init__(self):
        self.scene_patterns = [
            r'^(\d+-\d+[A-Z]?)\.(.*)$',
            r'^(\d+-\d+[A-Z]?)\s+(.*)$',
        ]
    
    def is_scene_tag(self, text: str) -> bool:
        return any(re.match(pattern, text) for pattern in self.scene_patterns)
    
    def parse_scene_tag(self, text: str) -> tuple:
        for pattern in self.scene_patterns:
            match = re.match(pattern, text)
            if match:
                scene_number = match.group(1)
                scene_title = match.group(2).strip()
                return scene_number, scene_title
        return None, None
    
    def parse_text(self, text: str) -> List[Dict]:
        scenes = []
        current_scene = None
        current_content = []
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if self.is_scene_tag(line):
                if current_scene and current_content:
                    current_scene['text'] = '\n'.join(current_content)
                    scenes.append(current_scene)
                    current_content = []
                
                scene_number, scene_title = self.parse_scene_tag(line)
                current_scene = {
                    'scene_number': scene_number,
                    'scene_title': scene_title,
                    'text': ''
                }
            else:
                if current_scene is not None:
                    current_content.append(line)
        
        if current_scene and current_content:
            current_scene['text'] = '\n'.join(current_content)
            scenes.append(current_scene)
        
        return scenes

class WordParser:
    def __init__(self):
        if Document is None:
            raise ImportError("python-docx не установлен")
    
    def parse(self, file_path: str) -> str:
        doc = Document(file_path)
        text_lines = []
        
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                text_lines.append(text)
        
        return '\n'.join(text_lines)

class PDFParser:
    def __init__(self):
        if PyPDF2 is None and pdfplumber is None:
            raise ImportError("PDF библиотеки не установлены")
    
    def parse_with_pypdf2(self, file_path: str) -> str:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def parse_with_pdfplumber(self, file_path: str) -> str:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    
    def parse(self, file_path: str) -> str:
        try:
            return self.parse_with_pdfplumber(file_path)
        except Exception:
            return self.parse_with_pypdf2(file_path)

class FileParser:
    def __init__(self):
        self.scene_parser = SceneParser()
        self.parsers = {}
        
        if Document is not None:
            self.parsers['docx'] = WordParser()
        
        if PyPDF2 is not None or pdfplumber is not None:
            self.parsers['pdf'] = PDFParser()
    
    def get_file_type(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.docx':
            return 'docx'
        elif ext == '.pdf':
            return 'pdf'
        else:
            raise ValueError(f"Неподдерживаемый формат: {ext}")
    
    def parse_file(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        file_type = self.get_file_type(file_path)
        
        if file_type not in self.parsers:
            raise ValueError(f"Парсер для {file_type} не доступен")
        
        text = self.parsers[file_type].parse(file_path)
        scenes = self.scene_parser.parse_text(text)
        
        return scenes

def save_to_json(scenes: List[Dict], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

def generate_json_filename(input_file_path: str) -> str:
    input_dir = os.path.dirname(input_file_path)
    input_filename = os.path.basename(input_file_path)
    base_name = os.path.splitext(input_filename)[0]
    json_filename = f"{base_name}_scenes.json"
    json_filepath = os.path.join(input_dir, json_filename)
    return json_filepath

def main():
    parser = argparse.ArgumentParser(description='Парсер сценариев', add_help=False)
    parser.add_argument('file_path', help='Путь к файлу')
    parser.add_argument('-o', '--output', help='Имя JSON файла')
    
    try:
        args = parser.parse_args()
        
        file_parser = FileParser()
        scenes = file_parser.parse_file(args.file_path)
        
        if not scenes:
            return
        
        if args.output:
            if os.path.isabs(args.output) or os.path.dirname(args.output):
                json_output = args.output
            else:
                input_dir = os.path.dirname(args.file_path)
                json_output = os.path.join(input_dir, args.output)
        else:
            json_output = generate_json_filename(args.file_path)
        
        save_to_json(scenes, json_output)
        
    except SystemExit:
        pass
    except Exception:
        pass

if __name__ == "__main__":
    main()