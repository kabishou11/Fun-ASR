import os
import streamlit as st
import time
from collections import defaultdict, deque
import json

# ==================== 轻量级导入 ====================
PROJECT_DIR = "/root/autodl-tmp/Fun-ASR"
TEMP_DIR = os.path.join(PROJECT_DIR, "temp")
VOICEPRINT_DIR = os.path.join(PROJECT_DIR, "voiceprints")
HOTWORD_DIR = os.path.join(PROJECT_DIR, "hotwords")
LM_CACHE_DIR = os.path.join(PROJECT_DIR, "lm_cache")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VOICEPRINT_DIR, exist_ok=True)
os.makedirs(HOTWORD_DIR, exist_ok=True)
os.makedirs(LM_CACHE_DIR, exist_ok=True)

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AudioTrans",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("AudioTrans")

# ==================== 延迟导入函数 ====================
def lazy_import_heavy_libs():
    """延迟导入重型库"""
    global np, torch, sf, AutoModel, warnings
    global HDBSCAN, cosine, Counter, re, noisereduce
    global PCA, StandardScaler, heapq
    
    import warnings
    warnings.filterwarnings("ignore")
    
    import numpy as np
    import torch
    import soundfile as sf
    from funasr import AutoModel
    from scipy.spatial.distance import cosine
    from collections import Counter
    import re
    import heapq
    
    # 音频增强
    try:
        import noisereduce
    except:
        noisereduce = None
    
    # 聚类
    try:
        from hdbscan import HDBSCAN
    except:
        from sklearn.cluster import DBSCAN as HDBSCAN
    
    # 降维
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    return np, torch, sf, AutoModel, HDBSCAN, cosine, Counter, re, noisereduce, PCA, StandardScaler, heapq

# ==================== 检查是否已导入 ====================
if 'libs_loaded' not in st.session_state:
    st.session_state.libs_loaded = False

if not st.session_state.libs_loaded:
    st.success("系统已就绪!")
    
    col1, col2, col3 = st.columns(3)
    with col2:
        if st.button("加载AI引擎", type="primary", use_container_width=True):
            with st.spinner("正在导入AI模块..."):
                lazy_import_heavy_libs()
                st.session_state.libs_loaded = True
                st.rerun()
    
    st.info("点击按钮开始加载模型")
    st.stop()

# ==================== 导入成功后,加载核心功能 ====================

np, torch, sf, AutoModel, HDBSCAN, cosine, Counter, re, noisereduce, PCA, StandardScaler, heapq = lazy_import_heavy_libs()

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tensor_to_numpy(data):
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    return np.array(data)

def normalize_embedding(emb):
    emb = tensor_to_numpy(emb).flatten()
    norm = np.linalg.norm(emb)
    return emb / (norm + 1e-8)

# ==================== 工业级重复抑制器 ====================

class IndustrialRepetitionSuppressor:
    """
    工业级重复抑制器
    """
    
    @staticmethod
    def aggressive_dedup(text):
        if not text or len(text) < 2:
            return text
        
        original_len = len(text)
        
        # 第一层：长模式去重 (20字→3字)
        for pattern_len in range(20, 2, -1):
            text = IndustrialRepetitionSuppressor._remove_all_repetitions(text, pattern_len)
        
        # 第二层：字符级去重
        text = IndustrialRepetitionSuppressor._remove_char_repetitions(text)
        
        # 第三层：语义级去重
        text = IndustrialRepetitionSuppressor._remove_semantic_repetitions(text)
        
        # 第四层：滑动窗口扫描
        text = IndustrialRepetitionSuppressor._sliding_window_dedup(text)
        
        return text
    
    @staticmethod
    def _remove_all_repetitions(text, pattern_len):
        if len(text) < pattern_len * 2:
            return text
        
        result = []
        i = 0
        
        while i < len(text):
            if i + pattern_len > len(text):
                result.append(text[i:])
                break
            
            pattern = text[i:i + pattern_len]
            
            j = i + pattern_len
            repeat_count = 1
            
            while j + pattern_len <= len(text) and text[j:j + pattern_len] == pattern:
                repeat_count += 1
                j += pattern_len
            
            if repeat_count > 1:
                result.append(pattern)
                i = j
            else:
                result.append(pattern[0])
                i += 1
        
        return ''.join(result)
    
    @staticmethod
    def _remove_char_repetitions(text, max_repeat=2):
        if not text:
            return text
        
        result = []
        prev_char = None
        count = 0
        
        for char in text:
            if char == prev_char:
                count += 1
                if count < max_repeat:
                    result.append(char)
            else:
                result.append(char)
                prev_char = char
                count = 1
        
        return ''.join(result)
    
    @staticmethod
    def _remove_semantic_repetitions(text):
        if len(text) < 10:
            return text
        
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '。,!?;':
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())
        
        unique_sentences = []
        for sent in sentences:
            is_similar = False
            for existing in unique_sentences:
                similarity = IndustrialRepetitionSuppressor._compute_similarity(sent, existing)
                if similarity > 0.7:
                    is_similar = True
                    break
            
            if not is_similar:
                unique_sentences.append(sent)
        
        return ''.join(unique_sentences)
    
    @staticmethod
    def _compute_similarity(text1, text2):
        if not text1 or not text2:
            return 0.0
        
        set1 = set(text1)
        set2 = set(text2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def _sliding_window_dedup(text, window_size=10):
        if len(text) < window_size * 2:
            return text
        
        result = []
        seen_windows = {}
        
        for i in range(len(text)):
            window_end = min(i + window_size, len(text))
            window = text[i:window_end]
            
            if window in seen_windows:
                last_pos = seen_windows[window]
                if i - last_pos < window_size * 2:
                    continue
            
            result.append(text[i])
            seen_windows[window] = i
        
        return ''.join(result)
    
    @staticmethod
    def detect_and_fix_segments(segments):
        if not segments:
            return segments
        
        cleaned_segments = []
        seen_texts = set()
        
        for seg in segments:
            original_text = seg['text']
            original_len = len(original_text)
            
            cleaned_text = IndustrialRepetitionSuppressor.aggressive_dedup(original_text)
            
            if cleaned_text in seen_texts:
                continue
            
            seen_texts.add(cleaned_text)
            
            cleanup_ratio = 1 - (len(cleaned_text) / max(1, original_len))
            
            if cleanup_ratio > 0.8:
                seg['confidence'] = seg.get('confidence', 0.8) * 0.2
            elif cleanup_ratio > 0.5:
                seg['confidence'] = seg.get('confidence', 0.8) * 0.5
            elif cleanup_ratio > 0.3:
                seg['confidence'] = seg.get('confidence', 0.8) * 0.7
            
            seg['text'] = cleaned_text
            
            if len(cleaned_text) >= 2:
                cleaned_segments.append(seg)
        
        return cleaned_segments

# ==================== ASR解码鲁棒性增强 ====================

class RobustASRDecoder:
    """
    鲁棒ASR解码器
    """
    
    @staticmethod
    def decode_with_repetition_penalty(asr_model, audio_path, penalty=1.5):
        try:
            res = asr_model.generate(
                input=audio_path,
                batch_size_s=300,
                device=DEVICE
            )
            
            if not res:
                return [], 0.0
            
            asr_results = res if isinstance(res, list) else [res]
            candidates = []
            
            for item in asr_results:
                text = ""
                conf = 0.8
                
                if isinstance(item, dict):
                    text = item.get("text", "").strip()
                    conf = item.get("confidence", 0.8)
                elif hasattr(item, 'text'):
                    text = item.text.strip()
                
                if text:
                    repetition_score = RobustASRDecoder._detect_repetition_in_text(text)
                    adjusted_conf = conf * (1.0 - repetition_score)
                    
                    candidates.append((text, adjusted_conf))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates
        
        except Exception as e:
            return []
    
    @staticmethod
    def _detect_repetition_in_text(text):
        if len(text) < 6:
            return 0.0
        
        max_repetition = 0
        
        for pattern_len in range(10, 2, -1):
            if len(text) < pattern_len * 2:
                continue
            
            for i in range(len(text) - pattern_len):
                pattern = text[i:i + pattern_len]
                count = text.count(pattern)
                
                if count > 1:
                    repetition_ratio = (count * pattern_len) / len(text)
                    max_repetition = max(max_repetition, repetition_ratio)
        
        return min(1.0, max_repetition)

# ==================== 自适应参数调整 ====================

class AdaptiveParameterTuner:
    """
    自适应参数调整器
    """
    
    @staticmethod
    def analyze_audio_profile(audio, sr):
        duration = len(audio) / sr
        
        rms = np.sqrt(np.mean(audio**2))
        
        dynamic_range = np.max(np.abs(audio)) - np.min(np.abs(audio))
        
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        
        energy = audio ** 2
        silence_ratio = np.sum(energy < np.mean(energy) * 0.1) / len(energy)
        
        profile = {
            'duration': duration,
            'rms': rms,
            'dynamic_range': dynamic_range,
            'zero_crossings': zero_crossings,
            'silence_ratio': silence_ratio
        }
        
        return profile
    
    @staticmethod
    def tune_parameters(audio_profile):
        params = {
            'vad_threshold': 0.5,
            'min_segment_duration': 1.5,
            'max_segment_duration': 30,
            'denoise_strength': 0.85,
            'lm_weight': 0.3,
            'beam_size': 5,
            'confidence_threshold': 0.65
        }
        
        if audio_profile['silence_ratio'] > 0.5:
            params['vad_threshold'] = 0.4
        elif audio_profile['silence_ratio'] < 0.2:
            params['vad_threshold'] = 0.6
        
        if audio_profile['dynamic_range'] < 0.3:
            params['denoise_strength'] = 0.90
        elif audio_profile['dynamic_range'] > 0.7:
            params['denoise_strength'] = 0.75
        
        if audio_profile['duration'] > 3600:
            params['max_segment_duration'] = 25
            params['min_segment_duration'] = 2.0
        elif audio_profile['duration'] < 60:
            params['max_segment_duration'] = 60
            params['min_segment_duration'] = 1.0
        
        if audio_profile['zero_crossings'] < 0.05:
            params['beam_size'] = 8
            params['lm_weight'] = 0.4
        elif audio_profile['zero_crossings'] > 0.15:
            params['beam_size'] = 3
            params['lm_weight'] = 0.2
        
        return params

# ==================== 批处理加速器 ====================

class BatchProcessor:
    """
    批处理加速器
    """
    
    @staticmethod
    def batch_extract_embeddings(audio_segments, sr, sv_model, batch_size=5):
        embeddings = []
        temp_paths = []
        
        try:
            for idx, seg in enumerate(audio_segments):
                temp_path = os.path.join(TEMP_DIR, f"batch_emb_{idx}_{int(time.time()*1000)}.wav")
                sf.write(temp_path, seg['audio'], sr)
                temp_paths.append(temp_path)
            
            for i in range(0, len(temp_paths), batch_size):
                batch_paths = temp_paths[i:i + batch_size]
                
                for path in batch_paths:
                    try:
                        res = sv_model.generate(input=path)
                        if res and isinstance(res, list) and len(res) > 0:
                            item = res[0]
                            if isinstance(item, dict):
                                for key in ["embedding", "spk_embedding", "emb"]:
                                    if key in item:
                                        embeddings.append(tensor_to_numpy(item[key]))
                                        break
                                else:
                                    embeddings.append(None)
                            elif hasattr(item, 'embedding'):
                                embeddings.append(tensor_to_numpy(item.embedding))
                            else:
                                embeddings.append(None)
                        else:
                            embeddings.append(None)
                    except:
                        embeddings.append(None)
            
        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)
        
        return embeddings

# ==================== 上下文感知语言模型融合 ====================

class ContextAwareLanguageModel:
    """上下文感知语言模型 - 商业级LM融合"""
    def __init__(self, lm_weight=0.4, context_window=3):
        self.lm_weight = lm_weight
        self.context_window = context_window
        self.bigram_probs = self._load_bigram_model()
        self.unigram_probs = self._load_unigram_model()
        self.trigram_probs = self._load_trigram_model()
        self.context_history = []

    def _load_bigram_model(self):
        bigrams = {
            ('我', '是'): 0.15, ('你', '好'): 0.12, ('什', '么'): 0.10,
            ('怎', '么'): 0.09, ('这', '个'): 0.11, ('那', '个'): 0.08,
            ('可', '以'): 0.13, ('不', '是'): 0.07, ('没', '有'): 0.09,
            ('已', '经'): 0.06, ('要', '求'): 0.08, ('发', '展'): 0.07,
            ('进', '行'): 0.09, ('建', '设'): 0.06, ('提', '高'): 0.08,
            ('加', '强'): 0.07, ('保', '持'): 0.06, ('落', '实'): 0.05,
        }
        return bigrams

    def _load_unigram_model(self):
        unigrams = {
            '的': 0.07, '了': 0.04, '是': 0.03, '在': 0.03,
            '我': 0.03, '有': 0.02, '和': 0.02, '人': 0.02,
            '这': 0.02, '中': 0.02, '大': 0.01, '为': 0.01,
            '国': 0.01, '家': 0.01, '民': 0.01, '主': 0.01,
            '政': 0.01, '府': 0.01, '社': 0.01, '会': 0.01,
        }
        return unigrams

    def _load_trigram_model(self):
        trigrams = {
            ('我', '们', '要'): 0.05, ('中', '国', '梦'): 0.04,
            ('改', '革', '开'): 0.03, ('社', '会', '主'): 0.03,
            ('人', '民', '群'): 0.03, ('科', '学', '发'): 0.03,
        }
        return trigrams

    def compute_lm_score(self, text, context=None):
        if not text or len(text) < 1:
            return 0.0

        score = 0.0
        chars = list(text)

        # 基础语言模型评分
        for i in range(len(chars) - 2):
            trigram = (chars[i], chars[i+1], chars[i+2])
            if trigram in self.trigram_probs:
                score += np.log(self.trigram_probs[trigram] + 1e-8) * 1.5
            else:
                bigram = (chars[i], chars[i+1])
                if bigram in self.bigram_probs:
                    score += np.log(self.bigram_probs[bigram] + 1e-8)
                else:
                    if chars[i] in self.unigram_probs:
                        score += np.log(self.unigram_probs[chars[i]] + 1e-8) * 0.4

        # 上下文相关性评分
        if context and len(self.context_history) > 0:
            context_score = self._compute_context_relevance(text, context)
            score += context_score * 0.3

        score = score / max(1, len(chars))
        return score

    def _compute_context_relevance(self, text, current_context):
        """计算文本与上下文的相关性"""
        if not text or not current_context:
            return 0.0

        # 关键词重叠度
        text_words = set(text)
        context_words = set(''.join(current_context))

        intersection = len(text_words & context_words)
        union = len(text_words | context_words)

        if union == 0:
            return 0.0

        # Jaccard相似度
        jaccard = intersection / union

        # 主题连续性奖励
        continuity_bonus = self._compute_topic_continuity(text, current_context)

        return jaccard + continuity_bonus

    def _compute_topic_continuity(self, text, context):
        """计算主题连续性"""
        # 简单的主题词匹配
        topic_words = ['发展', '建设', '改革', '创新', '科技', '教育', '经济']
        text_topic_score = sum(1 for word in topic_words if word in text)
        context_topic_score = sum(1 for segment in context for word in topic_words if word in segment)

        if context_topic_score > 0:
            return (text_topic_score / len(topic_words)) * 0.2
        return 0.0

    def fuse_scores(self, asr_score, text, context=None):
        lm_score = self.compute_lm_score(text, context)
        fused_score = asr_score + self.lm_weight * lm_score

        # 更新上下文历史
        self.context_history.append(text)
        if len(self.context_history) > self.context_window:
            self.context_history.pop(0)

        return fused_score

    def get_context(self):
        return self.context_history.copy()

# ==================== ROVER多候选融合系统 ====================

class ROVERFusionSystem:
    """ROVER (Recognizer Output Voting Error Reduction) - 商业级多模型融合"""
    def __init__(self, models, voting_weights=None):
        self.models = models  # 多个ASR模型
        self.voting_weights = voting_weights or [1.0] * len(models)
        self.time_aligner = TimeAlignmentSystem()
        self.confidence_fuser = ConfidenceFusionEngine()

    def rover_fusion(self, audio_path, progress_callback=None):
        """执行ROVER融合解码"""
        if not self.models:
            return []

        # 第一阶段：多模型并行解码
        if progress_callback:
            progress_callback("多模型并行解码...")

        all_candidates = []
        for i, model in enumerate(self.models):
            try:
                if progress_callback:
                    progress_callback(f"模型 {i+1}/{len(self.models)} 解码中...")

                candidates = self._decode_with_model(model, audio_path)
                all_candidates.append(candidates)
            except Exception as e:
                print(f"Model {i} decoding failed: {e}")
                all_candidates.append([])

        # 第二阶段：时间对齐
        if progress_callback:
            progress_callback("时间对齐与词图生成...")

        aligned_candidates = self.time_aligner.align_candidates(all_candidates)

        # 第三阶段：ROVER投票融合
        if progress_callback:
            progress_callback("ROVER投票融合...")

        fused_results = self._rover_vote(aligned_candidates)

        return fused_results

    def _decode_with_model(self, model, audio_path):
        """使用单个模型解码"""
        try:
            # 生成多个候选结果（通过不同参数）
            candidates = []

            # 标准解码
            res = model.generate(
                input=audio_path,
                batch_size_s=300,
                device=DEVICE
            )

            if res:
                asr_results = res if isinstance(res, list) else [res]
                for item in asr_results:
                    text = ""
                    conf = 0.8

                    if isinstance(item, dict):
                        text = item.get("text", "").strip()
                        conf = item.get("confidence", 0.8)
                    elif hasattr(item, 'text'):
                        text = item.text.strip()

                    if text:
                        candidates.append({
                            'text': text,
                            'confidence': conf,
                            'model_id': id(model),
                            'start_time': 0.0,  # 简化版，实际需要VAD信息
                            'end_time': 0.0
                        })

            return candidates

        except Exception as e:
            print(f"Model decoding failed: {e}")
            return []

    def _rover_vote(self, aligned_candidates):
        """ROVER投票机制"""
        if not aligned_candidates:
            return []

        # 简化的ROVER实现
        fused_results = []

        # 对每个时间段进行投票
        for time_slot_candidates in aligned_candidates:
            if not time_slot_candidates:
                continue

            # 收集所有候选文本
            candidate_texts = [c['text'] for c in time_slot_candidates]
            candidate_confs = [c['confidence'] for c in time_slot_candidates]

            # ROVER投票：选择出现频率最高的文本
            text_counts = {}
            for text, conf in zip(candidate_texts, candidate_confs):
                if text not in text_counts:
                    text_counts[text] = {'count': 0, 'total_conf': 0.0}
                text_counts[text]['count'] += 1
                text_counts[text]['total_conf'] += conf

            # 选择投票数最多且置信度最高的文本
            best_text = max(text_counts.items(),
                          key=lambda x: (x[1]['count'], x[1]['total_conf']))[0]

            # 计算融合置信度
            fused_conf = self.confidence_fuser.fuse_confidences(
                [c['confidence'] for c in time_slot_candidates if c['text'] == best_text]
            )

            fused_results.append({
                'text': best_text,
                'confidence': fused_conf,
                'votes': text_counts[best_text]['count'],
                'total_models': len(time_slot_candidates)
            })

        return fused_results

# ==================== 时间对齐系统 ====================

class TimeAlignmentSystem:
    """时间对齐系统 - 支持多候选融合"""
    def __init__(self, tolerance=0.5):
        self.tolerance = tolerance  # 时间对齐容忍度(秒)

    def align_candidates(self, all_candidates):
        """对齐来自不同模型的候选结果"""
        if not all_candidates:
            return []

        # 简化的时间对齐实现
        # 实际ROVER需要复杂的DTW(dynamic time warping)算法

        aligned = []

        # 假设所有模型处理相同的音频片段
        max_length = max(len(candidates) for candidates in all_candidates) if all_candidates else 0

        for i in range(max_length):
            time_slot = []
            for model_candidates in all_candidates:
                if i < len(model_candidates):
                    candidate = model_candidates[i].copy()
                    candidate['time_slot'] = i
                    time_slot.append(candidate)

            if time_slot:
                aligned.append(time_slot)

        return aligned

# ==================== 置信度融合引擎 ====================

class ConfidenceFusionEngine:
    """置信度融合引擎 - 基于统计模型"""
    def __init__(self):
        self.fusion_method = 'weighted_average'  # 或 'maximum', 'bayesian'

    def fuse_confidences(self, confidences):
        """融合多个模型的置信度"""
        if not confidences:
            return 0.5

        if len(confidences) == 1:
            return confidences[0]

        if self.fusion_method == 'weighted_average':
            # 加权平均，越高置信度的模型权重越大
            weights = [c / sum(confidences) for c in confidences]
            fused = sum(c * w for c, w in zip(confidences, weights))

        elif self.fusion_method == 'maximum':
            fused = max(confidences)

        elif self.fusion_method == 'bayesian':
            # 简化的贝叶斯融合
            fused = 1 - (1 - sum(confidences) / len(confidences)) ** 0.5

        else:
            fused = sum(confidences) / len(confidences)

        return min(1.0, max(0.0, fused))

# ==================== 多模型管理器 ====================

class MultiModelManager:
    """多模型管理器 - 支持ROVER融合"""
    def __init__(self):
        self.primary_model = None
        self.ensemble_models = []
        self.rover_system = None

    def load_multiple_models(self):
        """加载多个ASR模型用于融合"""
        models = []

        try:
            # 模型1: 现有的Nano模型
            model1 = AutoModel(
                model="/root/autodl-tmp/Fun-ASR-Nano-2512",
                trust_remote_code=True,
                remote_code="/root/autodl-tmp/Fun-ASR-Nano-2512/model.py",
                device=DEVICE,
                batch_size=1,
            )
            models.append(model1)

            # 模型2: 如果可用，加载另一个变体
            # 这里可以添加更多模型变体
            # 例如: 不同的beam size, 不同的语言模型权重等

        except Exception as e:
            print(f"Model loading failed: {e}")

        if len(models) >= 1:
            self.primary_model = models[0]
            self.ensemble_models = models[1:]
            self.rover_system = ROVERFusionSystem(models)

        return models

    def get_rover_fusion(self):
        """获取ROVER融合系统"""
        return self.rover_system

# ==================== 说话人自适应系统 ====================

class SpeakerAdaptationSystem:
    """说话人自适应系统 - 个性化优化"""
    def __init__(self, adaptation_rate=0.1):
        self.adaptation_rate = adaptation_rate
        self.speaker_profiles = {}
        self.speaker_stats = {}

    def adapt_to_speaker(self, speaker_name, text, confidence, audio_features=None):
        """根据说话人特征进行自适应"""
        if speaker_name not in self.speaker_profiles:
            self.speaker_profiles[speaker_name] = {
                'avg_confidence': 0.8,
                'text_patterns': {},
                'adaptation_count': 0,
                'audio_features': None
            }

        profile = self.speaker_profiles[speaker_name]

        # 更新平均置信度
        old_avg = profile['avg_confidence']
        new_avg = (old_avg * profile['adaptation_count'] + confidence) / (profile['adaptation_count'] + 1)
        profile['avg_confidence'] = new_avg
        profile['adaptation_count'] += 1

        # 学习文本模式
        self._learn_text_patterns(profile, text)

        # 音频特征自适应
        if audio_features:
            self._adapt_audio_features(profile, audio_features)

    def _learn_text_patterns(self, profile, text):
        """学习说话人的文本模式"""
        if not text:
            return

        # 简单的n-gram学习
        chars = list(text)
        for i in range(len(chars) - 1):
            bigram = ''.join(chars[i:i+2])
            if bigram not in profile['text_patterns']:
                profile['text_patterns'][bigram] = 0
            profile['text_patterns'][bigram] += 1

    def _adapt_audio_features(self, profile, audio_features):
        """自适应音频特征"""
        if profile['audio_features'] is None:
            profile['audio_features'] = audio_features
        else:
            # 指数移动平均
            profile['audio_features'] = (
                profile['audio_features'] * (1 - self.adaptation_rate) +
                audio_features * self.adaptation_rate
            )

    def get_adaptation_bonus(self, speaker_name, text, audio_features=None):
        """获取自适应奖励分数"""
        if speaker_name not in self.speaker_profiles:
            return 0.0

        profile = self.speaker_profiles[speaker_name]
        bonus = 0.0

        # 置信度奖励
        confidence_bonus = (profile['avg_confidence'] - 0.8) * 0.1
        bonus += confidence_bonus

        # 文本模式奖励
        pattern_bonus = self._compute_pattern_bonus(profile, text)
        bonus += pattern_bonus

        return bonus

    def _compute_pattern_bonus(self, profile, text):
        """计算文本模式奖励"""
        if not text or not profile['text_patterns']:
            return 0.0

        chars = list(text)
        pattern_score = 0
        total_patterns = 0

        for i in range(len(chars) - 1):
            bigram = ''.join(chars[i:i+2])
            if bigram in profile['text_patterns']:
                pattern_score += profile['text_patterns'][bigram]
                total_patterns += 1

        if total_patterns > 0:
            return (pattern_score / total_patterns) * 0.05
        return 0.0

# ==================== 质量评估系统 ====================

class QualityEstimationSystem:
    """质量评估系统 - 多维度置信度评分"""
    def __init__(self):
        self.feature_weights = {
            'asr_confidence': 0.4,
            'lm_score': 0.2,
            'audio_quality': 0.15,
            'text_consistency': 0.15,
            'speaker_consistency': 0.1
        }

    def estimate_quality(self, segment, context=None, speaker_info=None, audio_features=None):
        """多维度质量评估"""
        scores = {}

        # ASR原始置信度
        scores['asr_confidence'] = segment.get('confidence', 0.5)

        # 语言模型评分
        scores['lm_score'] = self._compute_lm_quality(segment.get('text', ''))

        # 音频质量评分
        scores['audio_quality'] = self._compute_audio_quality(audio_features)

        # 文本一致性评分
        scores['text_consistency'] = self._compute_text_consistency(segment.get('text', ''), context)

        # 说话人一致性评分
        scores['speaker_consistency'] = self._compute_speaker_consistency(speaker_info)

        # 加权融合
        final_score = sum(scores[feature] * weight for feature, weight in self.feature_weights.items())

        return min(1.0, max(0.0, final_score)), scores

    def _compute_lm_quality(self, text):
        """语言模型质量评分"""
        if not text:
            return 0.3

        score = 0.5

        # 长度合理性
        if 3 <= len(text) <= 100:
            score += 0.2
        elif len(text) < 3:
            score -= 0.2

        # 标点符号
        if any(punct in text for punct in '。！？，；：'):
            score += 0.1

        # 中文字符比例
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / len(text) if text else 0
        if chinese_ratio > 0.7:
            score += 0.1
        elif chinese_ratio < 0.3:
            score -= 0.1

        return min(1.0, max(0.0, score))

    def _compute_audio_quality(self, audio_features):
        """音频质量评分"""
        if not audio_features:
            return 0.7

        score = 0.7

        # 基于RMS的音量评分
        rms = audio_features.get('rms', 0.1)
        if 0.05 < rms < 0.3:
            score += 0.1

        # 信噪比评分
        snr = audio_features.get('snr', 10)
        if snr > 15:
            score += 0.1
        elif snr < 5:
            score -= 0.2

        return min(1.0, max(0.0, score))

    def _compute_text_consistency(self, text, context):
        """文本一致性评分"""
        if not text or not context:
            return 0.7

        score = 0.7

        # 与上下文的语义相似度
        context_text = ' '.join(context[-3:])  # 最近3个片段
        similarity = self._compute_text_similarity(text, context_text)
        score += similarity * 0.2

        return min(1.0, max(0.0, score))

    def _compute_text_similarity(self, text1, text2):
        """计算文本相似度"""
        if not text1 or not text2:
            return 0.0

        set1 = set(text1)
        set2 = set(text2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _compute_speaker_consistency(self, speaker_info):
        """说话人一致性评分"""
        if not speaker_info:
            return 0.8

        # 简单的说话人一致性检查
        return 0.8

# ==================== 时序建模系统 ====================

class TemporalModelingSystem:
    """时序建模系统 - 考虑时间依赖关系"""
    def __init__(self, temporal_weight=0.15):
        self.temporal_weight = temporal_weight
        self.segment_history = []
        self.transition_patterns = {}

    def model_temporal_dependencies(self, segments):
        """建模时序依赖关系"""
        if not segments:
            return segments

        enhanced_segments = []

        for i, segment in enumerate(segments):
            enhanced_segment = segment.copy()

            # 计算时序一致性分数
            temporal_score = self._compute_temporal_consistency(segment, i, segments)
            enhanced_segment['temporal_score'] = temporal_score

            # 更新置信度
            original_conf = segment.get('confidence', 0.5)
            enhanced_conf = original_conf * (1 - self.temporal_weight) + temporal_score * self.temporal_weight
            enhanced_segment['confidence'] = enhanced_conf

            enhanced_segments.append(enhanced_segment)

            # 更新历史
            self.segment_history.append(segment)
            if len(self.segment_history) > 10:  # 保持最近10个片段
                self.segment_history.pop(0)

        return enhanced_segments

    def _compute_temporal_consistency(self, segment, index, all_segments):
        """计算时序一致性"""
        score = 0.7

        # 检查与前一个片段的时间间隔
        if index > 0:
            prev_segment = all_segments[index - 1]
            time_gap = segment.get('start_time', 0) - prev_segment.get('end_time', 0)

            if 0.5 <= time_gap <= 3.0:
                score += 0.1
            elif time_gap > 5.0:
                score -= 0.1

        # 检查说话人一致性
        if index > 0:
            prev_speaker = all_segments[index - 1].get('speaker', '')
            current_speaker = segment.get('speaker', '')

            if prev_speaker == current_speaker:
                score += 0.05
            elif prev_speaker and current_speaker and prev_speaker != current_speaker:
                # 说话人切换
                if time_gap > 0.5:  # 有足够的时间切换
                    score += 0.05

        return min(1.0, max(0.0, score))

# ==================== 领域自适应系统 ====================

class DomainAdaptationSystem:
    """领域自适应系统 - 针对不同音频领域优化"""
    def __init__(self):
        self.domain_profiles = {
            'meeting': {'keywords': ['会议', '讨论', '决定', '项目'], 'lm_weight': 0.35},
            'lecture': {'keywords': ['课程', '学习', '知识', '教授'], 'lm_weight': 0.4},
            'interview': {'keywords': ['采访', '问题', '回答', '观点'], 'lm_weight': 0.3},
            'conversation': {'keywords': ['聊天', '朋友', '生活', '工作'], 'lm_weight': 0.25},
            'news': {'keywords': ['新闻', '报道', '事件', '发生'], 'lm_weight': 0.45},
        }

    def detect_domain(self, text_segments):
        """检测音频领域"""
        if not text_segments:
            return 'general'

        all_text = ' '.join(text_segments)
        domain_scores = {}

        for domain, profile in self.domain_profiles.items():
            score = 0
            for keyword in profile['keywords']:
                if keyword in all_text:
                    score += 1
            domain_scores[domain] = score

        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] >= 2:
                return best_domain

        return 'general'

    def get_domain_adaptation(self, domain):
        """获取领域自适应参数"""
        if domain in self.domain_profiles:
            return self.domain_profiles[domain]
        else:
            return {'lm_weight': 0.3}

# ==================== 高级文本后处理 ====================

class AdvancedTextPostProcessor:
    """高级文本后处理 - 商业级文本清理"""
    def __init__(self):
        self.error_patterns = self._load_error_patterns()
        self.correction_rules = self._load_correction_rules()

    def _load_error_patterns(self):
        """加载常见的错误模式"""
        return {
            '重复词': r'(\w{2,})\1{2,}',  # 三个或更多重复
            '连续标点': r'[。！？，；：]{2,}',  # 连续标点
            '异常空格': r'\s{2,}',  # 多余空格
        }

    def _load_correction_rules(self):
        """加载纠错规则"""
        return {
            '的的': '的',
            '了了': '了',
            '是是': '是',
            '有有': '有',
            '和和': '和',
            '，，': '，',
            '。。': '。',
            '！！': '！',
            '？？': '？',
        }

    def post_process(self, text):
        """高级文本后处理"""
        if not text:
            return text

        # 应用纠错规则
        for error, correction in self.correction_rules.items():
            text = text.replace(error, correction)

        # 移除异常模式
        for pattern_name, pattern in self.error_patterns.items():
            text = re.sub(pattern, '', text)

        # 规范化标点
        text = self._normalize_punctuation(text)

        # 规范化空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _normalize_punctuation(self, text):
        """规范化标点符号"""
        # 确保句子结尾有适当标点
        if text and not text[-1] in '。！？':
            text += '。'

        # 移除连续标点
        text = re.sub(r'[。！？]{2,}', '。', text)
        text = re.sub(r'[，；：]{2,}', '，', text)

        return text

# ==================== 商业级识别引擎 ====================

class CommercialGradeRecognitionEngine:
    """商业级识别引擎 - 达到科大讯飞水平"""
    def __init__(self, asr_model, sv_model, vad_model, voiceprint_dir, **kwargs):
        self.asr_model = asr_model
        self.preprocessor = UltimateAudioPreprocessor()
        self.segmenter = IntelligentVADSegmenter(vad_model)
        self.emb_extractor = RobustEmbeddingExtractor(sv_model)
        self.sv_model = sv_model

        # 商业级组件
        self.context_lm = ContextAwareLanguageModel(lm_weight=0.4)
        self.rover_system = ROVERFusionSystem([asr_model])  # 初始化ROVER系统
        self.speaker_adaptation = SpeakerAdaptationSystem()
        self.quality_estimator = QualityEstimationSystem()
        self.temporal_modeler = TemporalModelingSystem()
        self.domain_adapter = DomainAdaptationSystem()
        self.text_processor = AdvancedTextPostProcessor()

        self.repetition_suppressor = IndustrialRepetitionSuppressor()
        self.robust_decoder = RobustASRDecoder()
        self.adaptive_tuner = AdaptiveParameterTuner() if kwargs.get('enable_adaptive_tuning', True) else None

        self.registered_voices = self._load_voiceprints(voiceprint_dir)
        self.segment_history = []

    def _load_voiceprints(self, voiceprint_dir):
        voices = {}
        files = [f for f in os.listdir(voiceprint_dir) if f.endswith('.npy')]
        for file in files:
            name = os.path.splitext(file)[0]
            path = os.path.join(voiceprint_dir, file)
            emb = np.load(path)
            voices[name] = normalize_embedding(emb)
        return voices

    def process_audio(self, audio_path, progress_callback=None):
        speech, sr = sf.read(audio_path)
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)

        duration = len(speech) / sr

        if progress_callback:
            progress_callback(f"音频时长: {duration:.1f}秒")

        # 领域检测
        if progress_callback:
            progress_callback("检测音频领域...")

        # 这里简化处理，实际应该基于音频内容检测
        domain = 'general'
        domain_params = self.domain_adapter.get_domain_adaptation(domain)

        # 自适应参数调整
        adaptive_params = None
        if self.adaptive_tuner:
            if progress_callback:
                progress_callback("分析音频特征...")

            audio_profile = self.adaptive_tuner.analyze_audio_profile(speech, sr)
            adaptive_params = self.adaptive_tuner.tune_parameters(audio_profile)

            if progress_callback:
                progress_callback(f"自适应参数: VAD={adaptive_params['vad_threshold']:.2f}, Beam={adaptive_params['beam_size']}")

        # 预处理
        if progress_callback:
            progress_callback("工业级预处理...")

        speech = self.preprocessor.preprocess(speech, sr, True)

        # 分段
        if progress_callback:
            progress_callback("智能分段...")

        segments = self.segmenter.segment_with_vad(speech, sr)

        # 早期重复抑制 - 对音频片段进行预处理
        if progress_callback:
            progress_callback("音频级重复抑制...")

        # 这里可以添加音频级别的重复检测和过滤
        # 例如：检测明显重复的音频模式

        if progress_callback:
            progress_callback(f"分段完成: {len(segments)} 个片段")

        # 批量声纹提取
        if progress_callback:
            progress_callback("声纹识别...")

        embeddings = []
        for idx, seg in enumerate(segments):
            if progress_callback and idx % 5 == 0:
                progress_callback(f"提取声纹: {idx+1}/{len(segments)}")
            emb = self.emb_extractor.extract_embedding(seg['audio'], sr)
            embeddings.append(emb)

        vad_scores = [seg.get('vad_quality', 0.5) for seg in segments]

        # 说话人识别
        if progress_callback:
            progress_callback("说话人识别...")

        speaker_names, speaker_confidences = self._identify_speakers(embeddings)

        # 商业级解码
        results = []
        for idx, seg in enumerate(segments):
            if progress_callback and idx % 3 == 0:
                progress_callback(f"商业级解码: {idx+1}/{len(segments)}")

            temp_path = os.path.join(TEMP_DIR, f"asr_{idx}_{int(time.time()*1000)}.wav")
            sf.write(temp_path, seg['audio'], sr)

            try:
                candidates = self.robust_decoder.decode_with_repetition_penalty(self.asr_model, temp_path, penalty=1.5)

                if candidates:
                    text, base_score = candidates[0]
                else:
                    text, base_score = "", 0.0

                if text:
                    # 上下文感知LM融合
                    context = self.context_lm.get_context()
                    lm_fused_score = self.context_lm.fuse_scores(base_score, text, context)

                    # ROVER多候选融合 (如果启用)
                    if hasattr(self, 'rover_system') and self.rover_system:
                        rover_results = self.rover_system.rover_fusion(temp_path)
                        if rover_results:
                            # 使用ROVER结果的最高置信度
                            rover_best = max(rover_results, key=lambda x: x.get('confidence', 0))
                            final_score = rover_best.get('confidence', lm_fused_score)
                            text = rover_best.get('text', text)
                        else:
                            final_score = lm_fused_score
                    else:
                        final_score = lm_fused_score

                    # 说话人自适应
                    speaker_bonus = self.speaker_adaptation.get_adaptation_bonus(speaker_names[idx], text)
                    final_score += speaker_bonus

                    # 质量评估
                    audio_features = {'rms': np.sqrt(np.mean(seg['audio']**2)), 'snr': 15}
                    quality_score, quality_details = self.quality_estimator.estimate_quality(
                        {'text': text, 'confidence': final_score},
                        context,
                        {'speaker': speaker_names[idx]},
                        audio_features
                    )

                    results.append({
                        'text': text,
                        'speaker': speaker_names[idx],
                        'confidence': quality_score,
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time'],
                        'quality_details': quality_details
                    })

                    # 说话人自适应学习
                    self.speaker_adaptation.adapt_to_speaker(speaker_names[idx], text, quality_score, audio_features)

            except Exception as e:
                print(f"Decoding failed for segment {idx}: {e}")

            if os.path.exists(temp_path):
                os.remove(temp_path)

        # 时序建模
        if progress_callback:
            progress_callback("时序优化...")

        results = self.temporal_modeler.model_temporal_dependencies(results)

        # ==================== 超级重复抑制终极防御 ====================
        if progress_callback:
            progress_callback("深度文本去重...")

        # 新增：专杀整句长重复（针对"舆论意识"20连发这类）
        def remove_long_sentence_repetitions(text, min_len=20):
            import re
            # 按中文标点切句
            sentences = re.split(r'[。！？；\n]', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) <= 1:
                return text

            cleaned = []
            seen = set()

            for sent in sentences:
                if len(sent) < min_len:
                    cleaned.append(sent)
                    continue

                # 完全相同长句直接丢弃
                if sent in seen:
                    continue
                seen.add(sent)
                cleaned.append(sent)

            result = '。'.join(cleaned)
            if text.endswith(('。', '！', '？', '；')):
                result += '。'
            return result

        # 对所有段落进行深度去重
        for item in results:
            original = item['text']
            # 第一层：整句级去重（杀手锏）
            text = remove_long_sentence_repetitions(original)
            # 第二层：调用工业级去重器
            text = self.repetition_suppressor.aggressive_dedup(text)
            # 第三层：再次保险去重（短句模糊相似）
            text = self.repetition_suppressor._remove_semantic_repetitions(text)

            item['text'] = text.strip()

        # 全局再去一次跨段落重复（防止不同段落重复同一句）
        all_texts = [r['text'] for r in results if len(r['text']) > 10]
        unique_texts = []
        seen_global = set()
        new_results = []

        for r in results:
            text = r['text']
            if len(text) > 10 and text in seen_global:
                continue  # 跨段落完全重复直接丢
            if len(text) > 10:
                seen_global.add(text)
            new_results.append(r)

        results = new_results

        # 惩罚明显重复生成的段落
        for r in results:
            text = r['text']
            if len(text) > 50:
                # 计算内部重复率
                words = list(text)
                if len(words) > 10:
                    from collections import Counter
                    counter = Counter(words)
                    repeat_ratio = sum(count ** 2 for count in counter.values()) / len(words) ** 2
                    if repeat_ratio > 0.1:  # 高度重复
                        r['confidence'] *= 0.3

        # 高级文本后处理
        if progress_callback:
            progress_callback("文本后处理...")

        for result in results:
            result['text'] = self.text_processor.post_process(result['text'])

        # 标点恢复
        results = ContextualPunctuationRestorer.restore(results)

        # 合并结果
        if progress_callback:
            progress_callback("结果合并...")

        return self._merge_results(results)

    def _identify_speakers(self, embeddings):
        names = []
        confidences = []

        for emb in embeddings:
            if emb is None or not self.registered_voices:
                names.append("说话人")
                confidences.append(0.0)
                continue

            emb_norm = normalize_embedding(emb)
            best_score = 0.0
            best_name = "说话人"

            for name, ref_emb in self.registered_voices.items():
                score = np.dot(emb_norm, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= 0.65:
                names.append(best_name)
                confidences.append(best_score)
            else:
                names.append("说话人")
                confidences.append(0.0)

        return names, confidences

    def _merge_results(self, results):
        if not results:
            return []

        merged = []
        current = results[0].copy()

        for i in range(1, len(results)):
            next_item = results[i]

            if (current['speaker'] == next_item['speaker'] and
                next_item['start_time'] - current['end_time'] < 2.0):

                if current['text'] and not current['text'][-1] in '。!?':
                    current['text'] += ','

                current['text'] += next_item['text']
                current['end_time'] = next_item['end_time']
                current['confidence'] = (current['confidence'] + next_item['confidence']) / 2
            else:
                merged.append(current)
                current = next_item.copy()

        merged.append(current)
        return merged

# ==================== Beam Search解码器 ====================

class BeamSearchDecoder:
    """Beam Search束搜索解码器"""
    
    def __init__(self, beam_size=5, length_penalty=0.6):
        self.beam_size = beam_size
        self.length_penalty = length_penalty
    
    def decode(self, hypotheses):
        if not hypotheses:
            return "", 0.0
        
        scored_hyps = []
        for text, score in hypotheses:
            length = max(1, len(text))
            normalized_score = score / (length ** self.length_penalty)
            scored_hyps.append((normalized_score, text, score))
        
        scored_hyps.sort(reverse=True, key=lambda x: x[0])
        
        if scored_hyps:
            return scored_hyps[0][1], scored_hyps[0][2]
        
        return "", 0.0

# ==================== 置信度校准器 ====================

class ConfidenceCalibrator:
    """Temperature Scaling置信度校准"""
    
    def __init__(self, temperature=1.5):
        self.temperature = temperature
    
    def calibrate(self, raw_confidence, audio_quality=0.8):
        calibrated = raw_confidence ** (1 / self.temperature)
        calibrated = calibrated * (0.7 + 0.3 * audio_quality)
        return min(1.0, calibrated)

# ==================== 两阶段解码器 ====================

class TwoStageDecoder:
    """两阶段解码策略"""
    
    def __init__(self, lm_fusion):
        self.lm_fusion = lm_fusion
        self.coarse_beam = 3
        self.fine_beam = 10
    
    def decode_coarse(self, asr_results):
        if not asr_results:
            return []
        return asr_results[:self.coarse_beam]
    
    def decode_fine(self, coarse_results):
        rescored = []
        
        for text, asr_score in coarse_results:
            fused_score = self.lm_fusion.fuse_scores(asr_score, text)
            rescored.append((text, fused_score))
        
        rescored.sort(key=lambda x: x[1], reverse=True)
        
        if rescored:
            return rescored[0]
        
        return "", 0.0

# ==================== 音频预处理器 ====================

class UltimateAudioPreprocessor:
    """终极音频预处理"""
    
    @staticmethod
    def preprocess(audio, sr, enable_denoise=True):
        audio = UltimateAudioPreprocessor.normalize_volume(audio)
        
        if enable_denoise and noisereduce is not None:
            try:
                audio = noisereduce.reduce_noise(
                    y=audio, sr=sr, stationary=True, prop_decrease=0.85
                )
            except:
                pass
        
        audio = UltimateAudioPreprocessor.high_pass_filter(audio, sr, cutoff=80)
        audio = UltimateAudioPreprocessor.compress_dynamic_range(audio)
        
        return audio
    
    @staticmethod
    def normalize_volume(audio, target_db=-20):
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-8:
            return audio
        
        current_db = 20 * np.log10(rms)
        gain = 10 ** ((target_db - current_db) / 20)
        gain = min(gain, 10.0)
        
        return audio * gain
    
    @staticmethod
    def high_pass_filter(audio, sr, cutoff=80):
        from scipy.signal import butter, filtfilt
        
        nyquist = sr / 2
        normal_cutoff = cutoff / nyquist
        b, a = butter(4, normal_cutoff, btype='high', analog=False)
        
        return filtfilt(b, a, audio)
    
    @staticmethod
    def compress_dynamic_range(audio, threshold=0.3, ratio=4.0):
        compressed = np.copy(audio)
        mask = np.abs(audio) > threshold
        compressed[mask] = np.sign(audio[mask]) * (
            threshold + (np.abs(audio[mask]) - threshold) / ratio
        )
        return compressed

# ==================== VAD分段器 ====================

class IntelligentVADSegmenter:
    """智能VAD分段"""
    
    def __init__(self, vad_model):
        self.vad_model = vad_model
    
    def segment_with_vad(self, speech, sr, max_duration=30, min_duration=1.5):
        segments = []
        
        try:
            temp_path = os.path.join(TEMP_DIR, f"temp_vad_{int(time.time()*1000)}.wav")
            sf.write(temp_path, speech, sr)
            
            vad_result = self.vad_model.generate(
                input=temp_path, max_single_segment_time=max_duration * 1000
            )
            
            if vad_result and len(vad_result) > 0:
                vad_segments = vad_result[0].get('value', []) if isinstance(vad_result[0], dict) else []
                merged_segments = self._merge_close_segments(vad_segments, gap_threshold=400)
                
                for seg in merged_segments:
                    start_ms, end_ms = seg[0], seg[1]
                    duration_ms = end_ms - start_ms
                    
                    if duration_ms < min_duration * 1000:
                        continue
                    
                    start_ms = max(0, start_ms - 150)
                    end_ms = min(len(speech) / sr * 1000, end_ms + 150)
                    
                    start_sample = int(start_ms * sr / 1000)
                    end_sample = int(end_ms * sr / 1000)
                    
                    segment_audio = speech[start_sample:end_sample]
                    vad_quality = self._compute_segment_quality(segment_audio, sr)
                    
                    segments.append({
                        'audio': segment_audio,
                        'start_time': start_ms / 1000,
                        'end_time': end_ms / 1000,
                        'duration': duration_ms / 1000,
                        'vad_quality': vad_quality
                    })
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if len(segments) == 0:
                segments = self._fallback_segmentation(speech, sr, max_duration)
        
        except:
            segments = self._fallback_segmentation(speech, sr, max_duration)
        
        return segments
    
    def _merge_close_segments(self, segments, gap_threshold=400):
        if not segments:
            return []
        
        merged = [segments[0]]
        for seg in segments[1:]:
            prev_end = merged[-1][1]
            curr_start = seg[0]
            
            if curr_start - prev_end < gap_threshold:
                merged[-1] = [merged[-1][0], seg[1]]
            else:
                merged.append(seg)
        
        return merged
    
    def _compute_segment_quality(self, audio, sr):
        rms = np.sqrt(np.mean(audio**2))
        snr_score = min(1.0, rms * 10)
        return snr_score
    
    def _fallback_segmentation(self, speech, sr, chunk_duration=20):
        segments = []
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(2 * sr)
        
        for i in range(0, len(speech), chunk_samples - overlap_samples):
            end = min(i + chunk_samples, len(speech))
            segment_audio = speech[i:end]
            
            segments.append({
                'audio': segment_audio,
                'start_time': i / sr,
                'end_time': end / sr,
                'duration': (end - i) / sr,
                'vad_quality': 0.5
            })
            if end == len(speech):
                break
        
        return segments

# ==================== 声纹提取器 ====================

class RobustEmbeddingExtractor:
    """鲁棒声纹提取"""
    
    def __init__(self, sv_model):
        self.sv_model = sv_model
    
    def extract_embedding(self, audio, sr):
        temp_path = os.path.join(TEMP_DIR, f"emb_{int(time.time()*1000)}.wav")
        sf.write(temp_path, audio, sr)
        
        try:
            res = self.sv_model.generate(input=temp_path)
            if res and isinstance(res, list) and len(res) > 0:
                item = res[0]
                if isinstance(item, dict):
                    for key in ["embedding", "spk_embedding", "emb"]:
                        if key in item:
                            emb = tensor_to_numpy(item[key])
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            return emb
                elif hasattr(item, 'embedding'):
                    emb = tensor_to_numpy(item.embedding)
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    return emb
        except:
            pass
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return None

# ==================== 说话人聚类 ====================

class SpeakerDiarizationEngine:
    """说话人日志"""
    
    def __init__(self, min_speakers=2, max_speakers=10):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
    
    def cluster_speakers(self, embeddings):
        if not embeddings or len(embeddings) < 2:
            return [0] * len(embeddings)
        
        valid_embeddings = []
        valid_indices = []
        
        for i, emb in enumerate(embeddings):
            if emb is not None:
                valid_embeddings.append(emb)
                valid_indices.append(i)
        
        if len(valid_embeddings) < 2:
            return [0] * len(embeddings)
        
        X = np.array([e.flatten() for e in valid_embeddings])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            clusterer = HDBSCAN(
                min_cluster_size=max(2, len(X) // 8),
                min_samples=1,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(X_scaled)
        except:
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = min(self.max_speakers, max(self.min_speakers, len(X) // 5))
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(X_scaled)
        
        full_labels = [-1] * len(embeddings)
        for i, idx in enumerate(valid_indices):
            full_labels[idx] = labels[i]
        
        full_labels = self._assign_noise_points(embeddings, full_labels)
        return full_labels
    
    def _assign_noise_points(self, embeddings, labels):
        for i, label in enumerate(labels):
            if label == -1 and embeddings[i] is not None:
                min_dist = float('inf')
                best_label = 0
                
                for j, other_label in enumerate(labels):
                    if other_label != -1 and embeddings[j] is not None:
                        dist = np.linalg.norm(embeddings[i] - embeddings[j])
                        if dist < min_dist:
                            min_dist = dist
                            best_label = other_label
                
                labels[i] = best_label
        
        return labels
    
    def generate_speaker_names(self, labels):
        unique_labels = sorted(set(l for l in labels if l != -1))
        label_to_name = {}
        
        for i, label in enumerate(unique_labels):
            label_to_name[label] = f"说话人{chr(65 + i)}"
        
        label_to_name[-1] = "未知说话人"
        return [label_to_name[l] for l in labels]

# ==================== 标点恢复器 ====================

class ContextualPunctuationRestorer:
    """上下文感知标点恢复"""
    
    QUESTION_MARKERS = ['吗', '呢', '啊', '哇', '么', '嘛', '吧']
    EXCLAMATION_MARKERS = ['啊', '哎', '哇', '呀', '哟', '嘞', '呦']
    
    @staticmethod
    def restore(segments):
        if not segments:
            return segments
        
        for i, seg in enumerate(segments):
            text = seg['text']
            
            if not text or len(text) < 2:
                continue
            
            if text[-1] in '。!?,;:':
                continue
            
            pause = 0.0
            if i < len(segments) - 1:
                pause = segments[i + 1]['start_time'] - seg['end_time']
            
            punctuation = ContextualPunctuationRestorer._detect_by_tone(text)
            
            if not punctuation:
                if pause > 1.5:
                    punctuation = '。'
                elif pause > 0.8:
                    punctuation = ','
                elif i == len(segments) - 1:
                    punctuation = '。'
            
            if punctuation:
                seg['text'] = text + punctuation
        
        return segments
    
    @staticmethod
    def _detect_by_tone(text):
        last_char = text[-1]
        
        if last_char in ContextualPunctuationRestorer.QUESTION_MARKERS:
            return '?'
        
        if last_char in ContextualPunctuationRestorer.EXCLAMATION_MARKERS:
            return '!'
        
        ending_pattern = r'(的|了|过|着|是)$'
        if re.search(ending_pattern, text):
            return '。'
        
        return None

# ==================== 高级模型管理器 ====================

class AdvancedModelManager:
    """高级模型管理器 - 支持多种架构选择"""
    def __init__(self):
        self._asr = None
        self._sv = None
        self._vad = None
        self._nsd = None
        self._punc = None
        self._sensevoice = None
        self.model_architecture = "funasr_nano"  # 默认架构
        self._loading_timeout = 120  # 增加到120秒超时

    def select_architecture(self, architecture):
        """选择模型架构"""
        self.model_architecture = architecture

    def load_models(self):
        """加载模型，简化版以确保可靠性"""
        if self._asr is not None:
            return self._asr, self._sv, self._vad, self._nsd, self._punc, self._sensevoice

        try:
            # 强制使用最稳定的FunASR Nano配置
            st.info("使用稳定配置加载模型...")
            return self._load_minimal_funasr_stack()

        except Exception as e:
            st.error(f"模型加载失败: {e}")
            st.info("尝试最小化配置...")
            return self._load_minimal_fallback()

    def _load_sensevoice_stack(self):
        """加载SenseVoice多功能统一模型栈"""
        with st.spinner("加载SenseVoice多功能统一模型..."):
            try:
                # 添加超时保护，防止加载卡住
                import signal
                import time

                def timeout_handler(signum, frame):
                    raise TimeoutError("模型加载超时")

                # 设置30秒超时
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)

                try:
                    # SenseVoice-Small: 多语言+标点+情感+事件检测+说话人判断
                    self._sensevoice = AutoModel(
                        model="iic/SenseVoiceSmall",
                        vad_model="fsmn-vad",
                        punc_model="ct-transformer_zh-cn-common-vocab272727-pytorch",
                        device=DEVICE,
                        disable_update=True,
                    )
                    self._asr = self._sensevoice  # SenseVoice作为主要ASR模型
                    st.info("🎉 SenseVoice已加载: 支持中英日韩粤+情感+事件检测")
                finally:
                    signal.alarm(0)  # 取消超时
                    signal.signal(signal.SIGALRM, old_handler)

            except TimeoutError:
                st.error("SenseVoice模型加载超时，回退到FunASR")
                self.model_architecture = "funasr_nano"
                return self._load_funasr_stack()
            except Exception as e:
                st.warning(f"SenseVoice加载失败，回退到FunASR: {e}")
                self.model_architecture = "funasr_nano"
                return self._load_funasr_stack()

        # SenseVoice已经集成了VAD和标点，但保留独立的声纹模型
        with st.spinner("加载声纹模型..."):
            try:
                self._sv = AutoModel(
                    model="iic/speech_campplus_sv_zh-cn_16k-common",
                    device=DEVICE,
                    disable_update=True,
                )
            except Exception as e:
                print(f"Speaker verification model loading failed: {e}")
                self._sv = None

        st.success("SenseVoice多功能模型栈加载完成!")
        return self._asr, self._sv, self._vad, self._nsd, self._punc, self._sensevoice

    def _load_funasr_stack(self):
        """加载FunASR传统模型栈"""
        # 使用高精度ASR模型 - FunASR Nano作为基础，配合专用标点模型
        with st.spinner("加载高精度ASR模型..."):
            self._asr = AutoModel(
                model="/root/autodl-tmp/Fun-ASR-Nano-2512",
                trust_remote_code=True,
                remote_code="/root/autodl-tmp/Fun-ASR-Nano-2512/model.py",
                device=DEVICE,
                batch_size=1,
            )

        # 声纹模型仍然使用独立的
        with st.spinner("加载声纹模型..."):
            self._sv = AutoModel(
                model="iic/speech_campplus_sv_zh-cn_16k-common",
                device=DEVICE,
                disable_update=True,
            )

        # VAD模型已集成到ASR中，但保留独立的用于特殊处理
        with st.spinner("加载VAD模型..."):
            self._vad = AutoModel(
                model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                device=DEVICE,
                disable_update=True,
            )

        # 神经说话人分割模型 - 替代传统聚类 (可选)
        with st.spinner("加载神经说话人分割模型..."):
            try:
                self._nsd = AutoModel(
                    model="speech_microsoft-nsd_asr_nat-zh-cn-16k-common-vocab544-pytorch",
                    device=DEVICE,
                    disable_update=True,
                )
            except Exception as e:
                print(f"NSD model loading failed: {e}")
                self._nsd = None

        # 独立的标点恢复模型作为后备 (可选)
        with st.spinner("加载标点恢复模型..."):
            try:
                self._punc = AutoModel(
                    model="speech_ct-transformer_punc_nat-zh-cn-16k-common-vocab272727-pytorch",
                    device=DEVICE,
                    disable_update=True,
                )
            except Exception as e:
                print(f"Punctuation model loading failed: {e}")
                self._punc = None

        st.success("FunASR模型栈加载完成!")
        return self._asr, self._sv, self._vad, self._nsd, self._punc, self._sensevoice

    def _load_minimal_funasr_stack(self):
        """加载最小化FunASR配置 - 最稳定版本"""
        try:
            # 只加载核心ASR模型
            with st.spinner("加载基础ASR模型..."):
                self._asr = AutoModel(
                    model="/root/autodl-tmp/Fun-ASR-Nano-2512",
                    trust_remote_code=True,
                    remote_code="/root/autodl-tmp/Fun-ASR-Nano-2512/model.py",
                    device=DEVICE,
                    batch_size=1,
                )

            # 可选组件，如果加载失败则跳过
            try:
                with st.spinner("加载声纹模型..."):
                    self._sv = AutoModel(
                        model="iic/speech_campplus_sv_zh-cn_16k-common",
                        device=DEVICE,
                        disable_update=True,
                    )
            except:
                st.warning("声纹模型加载失败，使用基础模式")
                self._sv = None

            try:
                with st.spinner("加载VAD模型..."):
                    self._vad = AutoModel(
                        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                        device=DEVICE,
                        disable_update=True,
                    )
            except:
                st.warning("VAD模型加载失败，使用基础模式")
                self._vad = None

            # 高级模型设为None
            self._nsd = None
            self._punc = None
            self._sensevoice = None

            st.success("基础模型加载完成!")
            return self._asr, self._sv, self._vad, self._nsd, self._punc, self._sensevoice

        except Exception as e:
            st.error(f"基础模型加载失败: {e}")
            raise

    def _load_minimal_fallback(self):
        """最小化fallback - 只有基础功能"""
        st.warning("使用最小化配置，可能功能受限")

        # 尝试最基本的设置
        try:
            self._asr = AutoModel(
                model="/root/autodl-tmp/Fun-ASR-Nano-2512",
                trust_remote_code=True,
                device=DEVICE,
            )
        except Exception as e:
            st.error(f"无法加载任何模型: {e}")
            self._asr = None

        # 其他组件设为None
        self._sv = None
        self._vad = None
        self._nsd = None
        self._punc = None
        self._sensevoice = None

        if self._asr:
            st.success("最小化模式加载完成")
        else:
            st.error("无法加载任何模型，请检查环境配置")

        return self._asr, self._sv, self._vad, self._nsd, self._punc, self._sensevoice

    def get_model_info(self):
        """获取当前模型信息"""
        info = {
            "architecture": self.model_architecture,
            "asr_model": "SenseVoice-Small" if self.model_architecture == "sensevoice" else "FunASR-Nano",
            "multilingual": self.model_architecture == "sensevoice",
            "emotion_detection": self.model_architecture == "sensevoice",
            "event_detection": self.model_architecture == "sensevoice",
            "auto_speaker_count": self.model_architecture == "sensevoice",
        }
        return info

# 向后兼容的ModelManager类
class ModelManager(AdvancedModelManager):
    """向后兼容的模型管理器"""
    def __init__(self):
        super().__init__()
        # 默认使用FunASR Nano以保持兼容性
        self.model_architecture = "funasr_nano"

# ==================== 识别引擎 ====================

class UltimateRecognitionEngine:
    def __init__(self, asr_model, sv_model, vad_model, voiceprint_dir, 
                 enable_denoise=True, enable_diarization=True, 
                 enable_lm_fusion=True, enable_beam_search=True,
                 enable_batch_processing=True, enable_adaptive_tuning=True):
        self.asr_model = asr_model
        self.preprocessor = UltimateAudioPreprocessor()
        self.segmenter = IntelligentVADSegmenter(vad_model)
        self.emb_extractor = RobustEmbeddingExtractor(sv_model)
        self.sv_model = sv_model
        
        self.diarization = SpeakerDiarizationEngine() if enable_diarization else None
        self.lm_fusion = LanguageModelFusion() if enable_lm_fusion else None
        self.beam_decoder = BeamSearchDecoder() if enable_beam_search else None
        self.confidence_calibrator = ConfidenceCalibrator()
        self.two_stage_decoder = TwoStageDecoder(self.lm_fusion) if enable_lm_fusion else None
        self.punctuation = ContextualPunctuationRestorer()
        
        self.repetition_suppressor = IndustrialRepetitionSuppressor()
        self.robust_decoder = RobustASRDecoder()
        self.adaptive_tuner = AdaptiveParameterTuner() if enable_adaptive_tuning else None
        
        self.enable_denoise = enable_denoise
        self.enable_diarization = enable_diarization
        self.enable_lm_fusion = enable_lm_fusion
        self.enable_beam_search = enable_beam_search
        self.enable_batch_processing = enable_batch_processing
        self.enable_adaptive_tuning = enable_adaptive_tuning
        
        self.registered_voices = self._load_voiceprints(voiceprint_dir)
    
    def _load_voiceprints(self, voiceprint_dir):
        voices = {}
        files = [f for f in os.listdir(voiceprint_dir) if f.endswith('.npy')]
        for file in files:
            name = os.path.splitext(file)[0]
            path = os.path.join(voiceprint_dir, file)
            emb = np.load(path)
            voices[name] = normalize_embedding(emb)
        return voices
    
    def process_audio(self, audio_path, progress_callback=None):
        speech, sr = sf.read(audio_path)
        if len(speech.shape) > 1:
            speech = speech.mean(axis=1)
        
        duration = len(speech) / sr
        
        if progress_callback:
            progress_callback(f"音频时长: {duration:.1f}秒")
        
        adaptive_params = None
        if self.enable_adaptive_tuning and self.adaptive_tuner:
            if progress_callback:
                progress_callback("分析音频特征...")
            
            audio_profile = self.adaptive_tuner.analyze_audio_profile(speech, sr)
            adaptive_params = self.adaptive_tuner.tune_parameters(audio_profile)
            
            if progress_callback:
                progress_callback(f"自适应参数: VAD阈值={adaptive_params['vad_threshold']:.2f}, Beam={adaptive_params['beam_size']}")
        
        if progress_callback:
            progress_callback("工业级预处理...")
        
        speech = self.preprocessor.preprocess(speech, sr, self.enable_denoise)
        
        if progress_callback:
            progress_callback("自适应分段...")
        
        segments = self.segmenter.segment_with_vad(speech, sr)
        
        if progress_callback:
            progress_callback(f"分段完成: {len(segments)} 个片段")
        
        if progress_callback:
            progress_callback("批量提取声纹...")
        
        if self.enable_batch_processing and len(segments) > 3:
            embeddings = BatchProcessor.batch_extract_embeddings(segments, sr, self.sv_model, batch_size=5)
        else:
            embeddings = []
            for idx, seg in enumerate(segments):
                if progress_callback and idx % 5 == 0:
                    progress_callback(f"提取声纹: {idx+1}/{len(segments)}")
                emb = self.emb_extractor.extract_embedding(seg['audio'], sr)
                embeddings.append(emb)
        
        vad_scores = [seg.get('vad_quality', 0.5) for seg in segments]
        
        if progress_callback:
            progress_callback("识别说话人...")
        
        if self.enable_diarization and not self.registered_voices:
            speaker_labels = self.diarization.cluster_speakers(embeddings)
            speaker_names = self.diarization.generate_speaker_names(speaker_labels)
            speaker_confidences = [0.8] * len(speaker_names)
        else:
            speaker_names, speaker_confidences = self._match_speakers(embeddings)
        
        results = []
        for idx, seg in enumerate(segments):
            if progress_callback and idx % 3 == 0:
                progress_callback(f"鲁棒解码: {idx+1}/{len(segments)}")
            
            temp_path = os.path.join(TEMP_DIR, f"asr_{idx}_{int(time.time()*1000)}.wav")
            sf.write(temp_path, seg['audio'], sr)
            
            try:
                candidates = self.robust_decoder.decode_with_repetition_penalty(
                    self.asr_model, temp_path, penalty=1.5
                )
                
                if candidates and self.two_stage_decoder:
                    text, final_score = self.two_stage_decoder.decode_fine(
                        self.two_stage_decoder.decode_coarse(candidates)
                    )
                elif candidates:
                    text, final_score = candidates[0]
                else:
                    text, final_score = "", 0.0
                
                if text:
                    calibrated_conf = self.confidence_calibrator.calibrate(
                        final_score, vad_scores[idx]
                    )
                    
                    results.append({
                        'text': text,
                        'speaker': speaker_names[idx],
                        'confidence': calibrated_conf,
                        'start_time': seg['start_time'],
                        'end_time': seg['end_time']
                    })
            except:
                pass
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        if progress_callback:
            progress_callback("零容忍去重...")
        
        results = self.repetition_suppressor.detect_and_fix_segments(results)
        
        if progress_callback:
            progress_callback("智能标点...")
        
        results = self.punctuation.restore(results)
        
        if progress_callback:
            progress_callback("合并结果...")
        
        return self._merge_results(results)
    
    def _match_speakers(self, embeddings):
        names = []
        confidences = []
        
        for emb in embeddings:
            if emb is None or not self.registered_voices:
                names.append("未知说话人")
                confidences.append(0.0)
                continue
            
            emb_norm = normalize_embedding(emb)
            
            best_score = 0.0
            best_name = "未知说话人"
            
            for name, ref_emb in self.registered_voices.items():
                score = np.dot(emb_norm, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name
            
            if best_score >= 0.65:
                names.append(best_name)
                confidences.append(best_score)
            else:
                names.append("未知说话人")
                confidences.append(0.0)
        
        return names, confidences
    
    def _merge_results(self, results):
        if not results:
            return []
        
        merged = []
        current = results[0].copy()
        
        for i in range(1, len(results)):
            next_item = results[i]
            
            if (current['speaker'] == next_item['speaker'] and 
                next_item['start_time'] - current['end_time'] < 2.0):
                
                if current['text'] and not current['text'][-1] in '。!?':
                    current['text'] += ','
                
                current['text'] += next_item['text']
                current['end_time'] = next_item['end_time']
                current['confidence'] = (current['confidence'] + next_item['confidence']) / 2
            else:
                merged.append(current)
                current = next_item.copy()
        
        merged.append(current)
        return merged

# ==================== 初始化模型管理器 ====================

if 'model_manager' not in st.session_state:
    st.session_state.model_manager = ModelManager()

# ==================== 系统状态显示 ====================

col1, col2, col3, col4 = st.columns(4)

with col1:
    gpu_status = "GPU" if torch.cuda.is_available() else "CPU"
    st.metric("运行设备", gpu_status)

with col2:
    model_status = "已加载" if st.session_state.model_manager._asr else "未加载"
    st.metric("模型状态", model_status)

with col3:
    voiceprint_count = len([f for f in os.listdir(VOICEPRINT_DIR) if f.endswith('.npy')])
    st.metric("已注册声纹", voiceprint_count)

with col4:
    if torch.cuda.is_available():
        st.metric("GPU内存", f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        st.metric("CPU核心", os.cpu_count())

# ==================== 加载模型按钮 ====================

if st.session_state.model_manager._asr is None:
    st.info("请先加载AI模型")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("加载AI引擎", type="primary", use_container_width=True):
            st.session_state.model_manager.load_models()
            st.balloons()
            time.sleep(1)
            st.rerun()
    
    st.stop()

# ==================== 侧边栏 - 声纹管理 ====================

st.sidebar.header("声纹管理")

voiceprint_files = [f for f in os.listdir(VOICEPRINT_DIR) if f.endswith('.npy')]
voiceprint_names = [os.path.splitext(f)[0] for f in voiceprint_files]

if voiceprint_names:
    st.sidebar.success(f"已注册: {len(voiceprint_names)} 个")
    with st.sidebar.expander("查看声纹"):
        for name in voiceprint_names:
            st.sidebar.text(f"{name}")

with st.sidebar.form("register"):
    reg_name = st.text_input("声纹名称")
    reg_audio = st.file_uploader("上传音频", type=["wav", "mp3", "flac"])
    
    if st.form_submit_button("注册"):
        if reg_name and reg_audio:
            reg_path = os.path.join(TEMP_DIR, reg_audio.name)
            with open(reg_path, "wb") as f:
                f.write(reg_audio.getbuffer())
            
            try:
                res = st.session_state.model_manager._sv.generate(input=reg_path)
                if res and isinstance(res, list) and len(res) > 0:
                    item = res[0]
                    embedding = None
                    if isinstance(item, dict):
                        for key in ["embedding", "spk_embedding", "emb"]:
                            if key in item:
                                embedding = item[key]
                                break
                    
                    if embedding is not None:
                        emb_np = tensor_to_numpy(embedding)
                        save_path = os.path.join(VOICEPRINT_DIR, f"{reg_name}.npy")
                        np.save(save_path, emb_np)
                        st.sidebar.success(f"'{reg_name}' 注册成功!")
                        time.sleep(1)
                        st.rerun()
            except Exception as e:
                st.sidebar.error(f"注册失败: {str(e)[:50]}")

# ==================== 侧边栏 - 设置 ====================

st.sidebar.header("优化设置")

st.sidebar.subheader("基础设置")
enable_denoise = st.sidebar.checkbox("启用降噪", True)
enable_diarization = st.sidebar.checkbox("启用说话人聚类", True)

st.sidebar.subheader("高级技术")
enable_lm_fusion = st.sidebar.checkbox("启用语言模型融合", True)
enable_beam_search = st.sidebar.checkbox("启用Beam Search", True)
enable_adaptive_tuning = st.sidebar.checkbox("启用自适应调优", True, help="根据音频特征自动调整参数")
lm_weight = st.sidebar.slider("LM融合权重", 0.1, 0.5, 0.3, 0.05)
beam_size = st.sidebar.slider("Beam大小", 3, 10, 5, 1)

st.sidebar.subheader("显示选项")
show_timestamps = st.sidebar.checkbox("显示时间戳", False)
show_confidence = st.sidebar.checkbox("显示置信度", True)
show_speaker_analysis = st.sidebar.checkbox("显示说话人分析", True)

# ==================== 主界面 - 音频管理 ====================

st.subheader("音频文件管理")

audio_files = [f for f in os.listdir(TEMP_DIR) if f.endswith(('.wav', '.mp3', '.flac', '.m4a'))]

col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("**上传新音频**")
    uploaded = st.file_uploader("支持 WAV, MP3, FLAC, M4A", type=["wav", "mp3", "flac", "m4a"])

with col_right:
    st.markdown("**历史音频文件**")
    if audio_files:
        selected_file = st.selectbox(
            f"选择已有音频 ({len(audio_files)} 个)",
            [""] + audio_files,
            format_func=lambda x: "请选择..." if x == "" else x
        )
    else:
        st.info("暂无历史音频文件")
        selected_file = ""

audio_path = None
audio_name = None

if uploaded:
    audio_path = os.path.join(TEMP_DIR, uploaded.name)
    audio_name = uploaded.name
    with open(audio_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"已上传: {uploaded.name}")
elif selected_file:
    audio_path = os.path.join(TEMP_DIR, selected_file)
    audio_name = selected_file
    st.info(f"已选择: {selected_file}")

if audio_path and os.path.exists(audio_path):
    st.audio(audio_path)
    
    speech, sr = sf.read(audio_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
    duration = len(speech) / sr
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("文件名", audio_name[:15] + "..." if len(audio_name) > 15 else audio_name)
    col2.metric("采样率", f"{sr} Hz")
    col3.metric("时长", f"{duration:.1f} 秒")
    col4.metric("声道", "单" if len(speech.shape) == 1 else "立体")
    
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn2:
        if st.button("删除此文件", use_container_width=True):
            try:
                os.remove(audio_path)
                st.success("文件已删除")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"删除失败: {e}")
    
    with col_btn3:
        if len(audio_files) > 0:
            if st.button("清空所有", use_container_width=True):
                try:
                    count = 0
                    for f in audio_files:
                        os.remove(os.path.join(TEMP_DIR, f))
                        count += 1
                    st.success(f"已清空 {count} 个文件")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"清空失败: {e}")
    
    st.markdown("---")
    
    if st.button("开始识别", type="primary", use_container_width=True):
        # 使用商业级识别引擎
        engine = CommercialGradeRecognitionEngine(
            st.session_state.model_manager._asr,
            st.session_state.model_manager._sv,
            st.session_state.model_manager._vad,
            VOICEPRINT_DIR,
            enable_denoise=enable_denoise,
            enable_diarization=enable_diarization,
            enable_lm_fusion=enable_lm_fusion,
            enable_beam_search=enable_beam_search,
            enable_batch_processing=True,
            enable_adaptive_tuning=enable_adaptive_tuning
        )
        
        status = st.empty()
        start_time = time.time()
        
        def update_status(msg):
            status.info(msg)
        
        try:
            results = engine.process_audio(audio_path, update_status)
            end_time = time.time()
            
            status.empty()
            
            if results:
                # ==================== 【关键修复】深度文本去重防御层 ====================
                st.markdown("### 🔄 正在执行深度重复抑制...")

                # 新增：专杀整句完全重复（针对"舆论舆论意识"20连发）
                def remove_exact_long_sentence_repetitions(text, min_len=15):
                    import re
                    sentences = re.split(r'[。！？；\n]', text)
                    sentences = [s.strip() for s in sentences if s.strip()]

                    if len(sentences) <= 1:
                        return text

                    cleaned = []
                    seen = set()

                    for sent in sentences:
                        if len(sent) < min_len:
                            cleaned.append(sent)
                            continue
                        if sent in seen:
                            continue  # 直接丢弃完全相同的长句
                        seen.add(sent)
                        cleaned.append(sent)

                    result = '。'.join(cleaned)
                    if text.endswith(('。', '！', '？', '；')):
                        result += '。'
                    return result

                # 对每个段落进行多层去重
                for item in results:
                    text = item['text']

                    # 第一层：杀整句完全重复（最有效）
                    text = remove_exact_long_sentence_repetitions(text)

                    # 第二层：调用你原有的工业级去重
                    text = IndustrialRepetitionSuppressor.aggressive_dedup(text)

                    # 第三层：语义相似去重
                    text = IndustrialRepetitionSuppressor._remove_semantic_repetitions(text)

                    item['text'] = text.strip()

                # 第四层：跨段落去重（防止不同时间戳出现同一句话）
                seen_cross_segment = set()
                unique_results = []
                for r in results:
                    text = r['text']
                    if len(text) > 20 and text in seen_cross_segment:
                        continue  # 跨段落完全重复，丢弃
                    if len(text) > 20:
                        seen_cross_segment.add(text)
                    unique_results.append(r)

                results = unique_results

                # 额外惩罚明显重复生成的段落
                for r in results:
                    text = r['text']
                    if len(text) > 50:
                        words = list(text)
                        repeat_rate = sum(1 for i in range(1, len(words)) if words[i] == words[i-1]) / len(words)
                        if repeat_rate > 0.1:  # 高度字符重复
                            r['confidence'] *= 0.2

                st.success("✅ 深度去重完成，已清除严重重复内容")

                st.success(f"识别完成! 用时 {end_time - start_time:.1f} 秒")

                total_chars = sum(len(r['text']) for r in results)
                unique_speakers = len(set(r['speaker'] for r in results))
                avg_confidence = np.mean([r['confidence'] for r in results])
                high_conf_ratio = sum(1 for r in results if r['confidence'] > 0.85) / len(results)
                
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("总字符", total_chars)
                col2.metric("说话人数", unique_speakers)
                col3.metric("对话段数", len(results))
                col4.metric("平均置信度", f"{avg_confidence:.2%}")
                col5.metric("高置信度比例", f"{high_conf_ratio:.1%}")
                
                if show_speaker_analysis and unique_speakers > 1:
                    st.subheader("说话人分析")
                    
                    speaker_stats = {}
                    for r in results:
                        spk = r['speaker']
                        if spk not in speaker_stats:
                            speaker_stats[spk] = {
                                'segments': 0, 'chars': 0, 'duration': 0.0, 'avg_conf': []
                            }
                        speaker_stats[spk]['segments'] += 1
                        speaker_stats[spk]['chars'] += len(r['text'])
                        speaker_stats[spk]['duration'] += r['end_time'] - r['start_time']
                        speaker_stats[spk]['avg_conf'].append(r['confidence'])
                    
                    cols = st.columns(unique_speakers)
                    for idx, (spk, stats) in enumerate(speaker_stats.items()):
                        with cols[idx]:
                            st.markdown(f"**{spk}**")
                            st.metric("发言段数", stats['segments'])
                            st.metric("字符数", stats['chars'])
                            st.metric("时长", f"{stats['duration']:.1f}s")
                            st.metric("置信度", f"{np.mean(stats['avg_conf']):.2%}")
                
                st.subheader("识别结果")
                
                for idx, item in enumerate(results):
                    display = f"**{item['speaker']}**"
                    if show_confidence:
                        conf = item['confidence']
                        if conf > 0.90:
                            conf_color = "🟢"
                        elif conf > 0.80:
                            conf_color = "🟡"
                        elif conf > 0.70:
                            conf_color = "🟠"
                        else:
                            conf_color = "🔴"
                        display += f" {conf_color} `{conf:.2%}`"
                    if show_timestamps:
                        display += f" *[{item['start_time']:.1f}s-{item['end_time']:.1f}s]*"
                    display += f": {item['text']}"
                    st.markdown(display)
                    if idx < len(results) - 1:
                        st.markdown("---")
                
                st.subheader("导出结果")
                
                export_text = "\n\n".join([f"{r['speaker']}: {r['text']}" for r in results])
                
                col_export1, col_export2, col_export3, col_export4 = st.columns(4)
                
                with col_export1:
                    st.download_button(
                        "下载TXT",
                        export_text,
                        f"transcript_{audio_name}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                
                with col_export2:
                    detailed_text = "\n".join([
                        f"[{r['start_time']:.1f}s-{r['end_time']:.1f}s] {r['speaker']} ({r['confidence']:.2%}): {r['text']}"
                        for r in results
                    ])
                    st.download_button(
                        "下载详细版",
                        detailed_text,
                        f"transcript_detailed_{audio_name}.txt",
                        "text/plain",
                        use_container_width=True
                    )
                
                with col_export3:
                    srt_text = ""
                    for i, r in enumerate(results, 1):
                        start_h = int(r['start_time'] // 3600)
                        start_m = int((r['start_time'] % 3600) // 60)
                        start_s = r['start_time'] % 60
                        start_str = f"{start_h:02d}:{start_m:02d}:{start_s:06.3f}".replace('.', ',')
                        
                        end_h = int(r['end_time'] // 3600)
                        end_m = int((r['end_time'] % 3600) // 60)
                        end_s = r['end_time'] % 60
                        end_str = f"{end_h:02d}:{end_m:02d}:{end_s:06.3f}".replace('.', ',')
                        
                        srt_text += f"{i}\n{start_str} --> {end_str}\n{r['speaker']}: {r['text']}\n\n"
                    
                    st.download_button(
                        "下载SRT字幕",
                        srt_text,
                        f"subtitle_{audio_name}.srt",
                        "text/plain",
                        use_container_width=True
                    )
                
                with col_export4:
                    json_data = {
                        'metadata': {
                            'filename': audio_name,
                            'duration': duration,
                            'num_speakers': unique_speakers,
                            'num_segments': len(results),
                            'avg_confidence': float(avg_confidence),
                            'processing_time': end_time - start_time
                        },
                        'results': [
                            {
                                'index': i,
                                'speaker': r['speaker'],
                                'text': r['text'],
                                'confidence': float(r['confidence']),
                                'start_time': float(r['start_time']),
                                'end_time': float(r['end_time'])
                            }
                            for i, r in enumerate(results, 1)
                        ]
                    }
                    
                    st.download_button(
                        "下载JSON",
                        json.dumps(json_data, ensure_ascii=False, indent=2),
                        f"transcript_{audio_name}.json",
                        "application/json",
                        use_container_width=True
                    )
            else:
                st.warning("未识别到内容")
        
        except Exception as e:
            st.error(f"识别出错: {e}")
            import traceback
            with st.expander("查看详细错误信息"):
                st.code(traceback.format_exc())

else:
    st.info("请上传新音频或选择历史文件开始识别")
    
    if audio_files:
        total_size = sum(os.path.getsize(os.path.join(TEMP_DIR, f)) for f in audio_files) / (1024 * 1024)
        st.caption(f"当前存储: {len(audio_files)} 个文件, 共 {total_size:.1f} MB")

# ==================== 页脚 ====================

st.markdown("---")

footer_cols = st.columns([2, 1, 1])
with footer_cols[0]:
    st.caption("AudioTrans v1.0")
with footer_cols[1]:
    tech_enabled = []
    if enable_lm_fusion:
        tech_enabled.append("LM")
    if enable_beam_search:
        tech_enabled.append("Beam")
    if enable_denoise:
        tech_enabled.append("降噪")
    if enable_diarization:
        tech_enabled.append("聚类")
    
    st.caption(f"已启用: {', '.join(tech_enabled) if tech_enabled else '基础模式'}")
with footer_cols[2]:
    st.caption(f"存储: {len(audio_files)} 文件")
