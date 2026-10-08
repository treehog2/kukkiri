from __future__ import annotations

LANGUAGES = {"en": "English", "ko": "한국어", "zh_CN": "简体中文", "zh_TW": "繁體中文"}

_STRINGS = {
    "Add videos…": ("동영상 추가…", "添加视频…", "新增影片…"),
    "Add folder…": ("폴더 추가…", "添加文件夹…", "新增資料夾…"),
    "Remove": ("제거", "移除", "移除"),
    "Advanced…": ("고급…", "高级…", "進階…"),
    "Before/after previews": ("전후 비교 미리보기", "前后对比预览", "前後對比預覽"),
    "Also write up to 3 short side-by-side clips (with a kukkiri watermark) of the most restored moments, next to "
    "each restored video": (
        "복원된 동영상 옆에, 가장 많이 복원된 장면의 짧은 비교 클립(kukkiri 워터마크 포함)을 최대 3개까지 함께 저장합니다",
        "在每个修复后的视频旁，另存最多 3 段修复最多片段的并排对比短片（带 kukkiri 水印）",
        "在每部修復後的影片旁，另存最多 3 段修復最多片段的並排對比短片（含 kukkiri 浮水印）",
    ),
    "Start": ("시작", "开始", "開始"),
    "Stop": ("중지", "停止", "停止"),
    "Log": ("로그", "日志", "記錄"),
    "System check": ("시스템 점검", "系统检查", "系統檢查"),
    "Language": ("언어", "语言", "語言"),
    "Browse…": ("찾아보기…", "浏览…", "瀏覽…"),
    "Choose folder": ("폴더 선택", "选择文件夹", "選擇資料夾"),
    "Add videos": ("동영상 추가", "添加视频", "新增影片"),
    "Videos": ("동영상", "视频", "影片"),
    "Add a folder of videos": ("동영상 폴더 추가", "添加视频文件夹", "新增影片資料夾"),
    "Model": ("모델", "模型", "模型"),
    "Weights folder": ("모델 파일 폴더", "模型文件夹", "模型資料夾"),
    "Output folder": ("출력 폴더", "输出文件夹", "輸出資料夾"),
    "next to each input": ("각 원본 파일 옆", "与原视频相同位置", "與原影片相同位置"),
    "latest": ("최신", "最新", "最新"),
    "Detection sensitivity": ("감지 민감도", "检测灵敏度", "偵測靈敏度"),
    "Normal": ("보통", "普通", "普通"),
    "High (also finds faint or small mosaics)": (
        "높음 (흐리거나 작은 모자이크도 감지)", "高（也能检测模糊或细小的马赛克）", "高（也能偵測模糊或細小的馬賽克）"),
    "Low (fewer false detections)": ("낮음 (오감지 감소)", "低（减少误检）", "低（減少誤判）"),
    "Limit to 1080p (faster for 4K videos)": ("1080p로 제한 (4K 동영상이 더 빨라짐)", "限制为 1080p（4K 视频更快）",
                                              "限制為 1080p（4K 影片更快）"),
    "Restore even if a video looks uncensored": ("무수정으로 보이는 동영상도 복원", "即使视频看起来没有打码也进行修复",
                                                 "即使影片看起來沒有打碼也進行修復"),
    "File": ("파일", "文件", "檔案"),
    "Section": ("구간", "片段", "片段"),
    "Status": ("상태", "状态", "狀態"),
    "Progress": ("진행률", "进度", "進度"),
    "Speed": ("속도", "速度", "速度"),
    "Time left": ("남은 시간", "剩余时间", "剩餘時間"),
    "{n} min": ("{n}분", "{n} 分钟", "{n} 分鐘"),
    "whole video": ("전체 동영상", "整个视频", "整部影片"),
    "from {start}": ("{start}부터", "从 {start} 开始", "從 {start} 開始"),
    "Restore only part of this video": ("이 동영상의 일부만 복원", "只修复此视频的一部分", "只修復此影片的一部分"),
    "Duration": ("길이", "时长", "時長"),
    "from the beginning (e.g. 37:45)": ("처음부터 (예: 37:45)", "从头开始（例如 37:45）", "從頭開始（例如 37:45）"),
    "to the end (e.g. 30)": ("끝까지 (예: 30)", "到结尾（例如 30）", "到結尾（例如 30）"),
    "Whole video": ("전체 동영상", "整个视频", "整部影片"),
    "OK": ("확인", "确定", "確定"),
    "Cancel": ("취소", "取消", "取消"),
    "Times are seconds, mm:ss or hh:mm:ss.": (
        "시간은 초, mm:ss 또는 hh:mm:ss 형식으로 입력하세요.",
        "时间格式为秒数、mm:ss 或 hh:mm:ss。",
        "時間格式為秒數、mm:ss 或 hh:mm:ss。",
    ),
    "Waiting": ("대기 중", "等待中", "等待中"),
    "Starting": ("시작 중", "正在启动", "正在啟動"),
    "Restoring": ("복원 중", "正在修复", "正在修復"),
    "Finishing": ("마무리 중", "正在完成", "正在完成"),
    "Preparing": ("준비 중", "正在准备", "正在準備"),
    "Downloading weights": ("모델 다운로드 중", "正在下载模型", "正在下載模型"),
    "Checking": ("확인 중", "正在检查", "正在檢查"),
    "Cutting section": ("구간 자르는 중", "正在截取片段", "正在擷取片段"),
    "Cancelled": ("취소됨", "已取消", "已取消"),
    "Done": ("완료", "完成", "完成"),
    "Skipped: {reason}": ("건너뜀: {reason}", "已跳过：{reason}", "已略過：{reason}"),
    "nothing to restore": ("복원할 부분 없음", "没有需要修复的内容", "沒有需要修復的內容"),
    "already uncensored or restored": ("이미 무수정이거나 복원됨", "已无码或已修复", "已無碼或已修復"),
    "Failed (exit {code}), see log": ("실패 (코드 {code}), 로그 참조", "失败（代码 {code}），请查看日志", "失敗（代碼 {code}），請查看記錄"),
    "Add videos or drop them here. Double-click a finished row to open it.": (
        "동영상을 추가하거나 여기로 끌어다 놓으세요. 완료된 행을 더블클릭하면 열립니다.",
        "添加视频或拖放到此处。双击已完成的行即可打开。",
        "新增影片或拖放到此處。按兩下已完成的列即可開啟。",
    ),
    "Stopped.": ("중지됨.", "已停止。", "已停止。"),
    "Queue finished.": ("대기열 완료.", "队列已完成。", "佇列已完成。"),
    "kukkiri needs the {model} model weights (about 240 MB). Download them now into\n{weights}?": (
        "kukkiri에 {model} 모델 파일(약 240 MB)이 필요합니다. 지금 다음 위치로 다운로드할까요?\n{weights}",
        "kukkiri 需要 {model} 模型文件（约 240 MB）。现在下载到以下位置吗？\n{weights}",
        "kukkiri 需要 {model} 模型檔案（約 240 MB）。現在下載到以下位置嗎？\n{weights}",
    ),
    "A video is still being restored. Stop it and quit?": (
        "아직 동영상을 복원하고 있습니다. 중지하고 종료할까요?",
        "仍有视频正在修复。要停止并退出吗？",
        "仍有影片正在修復。要停止並結束嗎？",
    ),
    "Checking…": ("확인 중…", "正在检查…", "正在檢查…"),
    "Check again": ("다시 확인", "重新检查", "重新檢查"),
    "Close": ("닫기", "关闭", "關閉"),
    "The check could not run (exit {code}):": ("점검을 실행할 수 없습니다 (코드 {code}):", "无法运行检查（代码 {code}）：", "無法執行檢查（代碼 {code}）："),
    "Graphics card": ("그래픽 카드", "显卡", "顯示卡"),
    "NVIDIA driver": ("NVIDIA 드라이버", "NVIDIA 驱动程序", "NVIDIA 驅動程式"),
    "GPU memory": ("GPU 메모리", "显存", "顯示記憶體"),
    "Free disk space": ("디스크 여유 공간", "可用磁盘空间", "可用磁碟空間"),
    "Model files ({model})": ("모델 파일 ({model})", "模型文件（{model}）", "模型檔案（{model}）"),
    "GPU preparation": ("GPU 준비", "GPU 准备", "GPU 準備"),
    "no NVIDIA GPU found": ("NVIDIA GPU를 찾을 수 없음", "未找到 NVIDIA 显卡", "找不到 NVIDIA 顯示卡"),
    "{name}: needs a GTX 16-series / RTX 20-series or newer": (
        "{name}: GTX 16 시리즈 / RTX 20 시리즈 이상이 필요합니다",
        "{name}：需要 GTX 16 系列 / RTX 20 系列或更新的显卡",
        "{name}：需要 GTX 16 系列 / RTX 20 系列或更新的顯示卡",
    ),
    "{version}: update to 580 or newer (nvidia.com/drivers)": (
        "{version}: 580 이상으로 업데이트하세요 (nvidia.com/drivers)",
        "{version}：请更新到 580 或更高版本（nvidia.com/drivers）",
        "{version}：請更新到 580 或更新版本（nvidia.com/drivers）",
    ),
    "{free} of {total} GB free": ("{total} GB 중 {free} GB 사용 가능", "{total} GB 中可用 {free} GB", "{total} GB 中可用 {free} GB"),
    "{free} of {total} GB free: close other programs using the GPU": (
        "{total} GB 중 {free} GB 사용 가능: GPU를 사용하는 다른 프로그램을 닫으세요",
        "{total} GB 中可用 {free} GB：请关闭其他占用显卡的程序",
        "{total} GB 中可用 {free} GB：請關閉其他使用顯示卡的程式",
    ),
    "{free} GB free on {path}": ("{path}에 {free} GB 여유", "{path} 可用 {free} GB", "{path} 可用 {free} GB"),
    "{free} GB free on {path}: free some space before restoring": (
        "{path}에 {free} GB 여유: 복원 전에 공간을 확보하세요",
        "{path} 可用 {free} GB：请先释放空间",
        "{path} 可用 {free} GB：請先釋放空間",
    ),
    "downloaded and verified": ("다운로드 및 확인 완료", "已下载并验证", "已下載並驗證"),
    "not downloaded yet: the first restore downloads them (about 240 MB)": (
        "아직 다운로드하지 않음: 첫 복원 시 다운로드됩니다 (약 240 MB)",
        "尚未下载：首次修复时会自动下载（约 240 MB）",
        "尚未下載：首次修復時會自動下載（約 240 MB）",
    ),
    "damaged: delete {path} and restore again to download them anew": (
        "손상됨: {path} 폴더를 삭제한 뒤 다시 복원하면 새로 다운로드됩니다",
        "已损坏：删除 {path} 后再次修复即可重新下载",
        "已損壞：刪除 {path} 後再次修復即可重新下載",
    ),
    "not found": ("찾을 수 없음", "未找到", "找不到"),
    "done": ("완료", "已完成", "已完成"),
    "not yet: the first restore takes 10 to 20 minutes longer": (
        "아직 안 됨: 첫 복원은 10~20분 더 걸립니다",
        "尚未完成：首次修复会多花 10 到 20 分钟",
        "尚未完成：首次修復會多花 10 到 20 分鐘",
    ),
}
_COLUMN = {"ko": 0, "zh_CN": 1, "zh_TW": 2}
_lang = "en"


def system_language(locale_name: str) -> str:
    if locale_name.startswith("ko"):
        return "ko"
    if locale_name.startswith("zh"):
        return "zh_TW" if any(t in locale_name for t in ("TW", "HK", "MO", "Hant")) else "zh_CN"
    return "en"


def set_language(lang: str) -> None:
    global _lang
    _lang = lang if lang in LANGUAGES else "en"


def language() -> str:
    return _lang


def tr(key: str, **kw) -> str:
    row = _STRINGS.get(key)
    text = row[_COLUMN[_lang]] if row and _lang in _COLUMN else key
    return text.format(**kw) if kw else text
