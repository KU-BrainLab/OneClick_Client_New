# -*- coding:utf-8 -*-
"""matplotlib 버전 차이 흡수.

cm.get_cmap 은 matplotlib 3.7 에서 폐기되고 3.9 에서 제거됐다. 측정 PC 와
분석 서버의 matplotlib 버전이 다를 수 있어(3.9 미만/이상 혼재) 콜맵 조회를
여기 한 곳에서 흡수한다.
"""
import matplotlib


def get_cmap(name):
    """이름으로 콜맵을 얻는다.

    등록된 원본 객체를 돌려주므로 고쳐 쓸 거면 .copy() 할 것.
    """
    try:
        return matplotlib.colormaps[name]          # 3.5+
    except AttributeError:                         # 그 이전 버전
        from matplotlib import cm
        return cm.get_cmap(name)
