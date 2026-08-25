import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([0,1,2],[0,1,0])
plt.title("Haz 3 clicks y luego Enter")
plt.show(block=False)

pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
print("Puntos capturados:", pts)