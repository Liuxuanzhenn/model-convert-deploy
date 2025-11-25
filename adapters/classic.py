"""传统机器学习适配器

合并所有classic_xxx.py文件
"""
import os
from typing import Iterable, List
from .base import ModelAdapter
from .registry import register


@register("sklearn", "dbscan")
@register("traditional_ml", "dbscan")
class ClassicDBSCANAdapter(ModelAdapter):
    """DBSCAN模型适配器"""

    def load(self) -> None:
        """加载DBSCAN模型"""
        weight = self._find_weight(extensions=(".pkl", ".joblib"))
        if not weight:
            self.model = None
            return

        try:
            import joblib
            self.model = joblib.load(weight)
        except ImportError:
            try:
                import pickle
                with open(weight, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None
        except Exception:
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出DBSCAN模型"""
        out = []
        if self.model is None:
            return out

        if "pickle" in formats or "pkl" in formats:
            try:
                import pickle
                path = os.path.join(self.artifacts_dir, "dbscan_model.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                out.append(path)
            except Exception:
                pass

        return out


@register("sklearn", "kmeans")
@register("traditional_ml", "kmeans")
class ClassicKMeansAdapter(ModelAdapter):
    """K-Means模型适配器"""

    def load(self) -> None:
        """加载K-Means模型"""
        weight = self._find_weight(extensions=(".pkl", ".joblib"))
        if not weight:
            self.model = None
            return

        try:
            import joblib
            self.model = joblib.load(weight)
        except ImportError:
            try:
                import pickle
                with open(weight, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None
        except Exception:
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出K-Means模型"""
        out = []
        if self.model is None:
            return out

        if "pickle" in formats or "pkl" in formats:
            try:
                import pickle
                path = os.path.join(self.artifacts_dir, "kmeans_model.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                out.append(path)
            except Exception:
                pass

        return out


@register("sklearn", "spectral_clustering")
@register("sklearn", "sc")
@register("traditional_ml", "spectral_clustering")
@register("traditional_ml", "sc")
class ClassicSpectralClusteringAdapter(ModelAdapter):
    """Spectral Clustering模型适配器"""

    def load(self) -> None:
        """加载Spectral Clustering模型"""
        weight = self._find_weight(extensions=(".pkl", ".joblib"))
        if not weight:
            self.model = None
            return

        try:
            import joblib
            self.model = joblib.load(weight)
        except ImportError:
            try:
                import pickle
                with open(weight, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception:
                self.model = None
        except Exception:
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出Spectral Clustering模型"""
        out = []
        if self.model is None:
            return out

        if "pickle" in formats or "pkl" in formats:
            try:
                import pickle
                path = os.path.join(self.artifacts_dir, "spectral_clustering_model.pkl")
                with open(path, 'wb') as f:
                    pickle.dump(self.model, f)
                out.append(path)
            except Exception:
                pass

        return out

