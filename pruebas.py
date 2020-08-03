import pyqtcss

print(pyqtcss.available_styles())
    #['classic', 'dark_blue', 'dark_orange']
    
style_string = pyqtcss.get_style("dark_blue")
qt_widget.setStyleSheet(style_string)