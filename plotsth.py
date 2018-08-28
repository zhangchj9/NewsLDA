import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

Month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# from pylab import mpl
# mpl.rcParams['font.sans-serif'] = ['YaHei Consolas Hybrid']
report_num = [13, 2,4,9,16,9,20,3,19,2,33,6]

for a,b in zip(Month,report_num):
    plt.text(a,b, b, ha='center', va='bottom', fontsize=12)

plt.plot(Month, report_num, '-o', color='orange', markersize=6)
plt.grid(Month, color='grey',linestyle='dashed')
plt.ylabel('Report Number')
plt.xlabel('Month')
ax = plt.gca()
# ax.xaxis.set_major_locator(MultipleLocator(3))
# ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))
# plt.xlim(0,12)
plt.ylim(0, 40)
plt.show()