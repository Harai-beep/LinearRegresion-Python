import numpy as np
import matplotlib.pyplot as plt
 
def estimate_coef(x, y):
    # numarul de observatii 
    n = np.size(x)
 
    # media vectorilor x si y
    m_x = np.mean(x)
    m_y = np.mean(y)
 
    # deviatia pt x
    SS_xy = np.sum(y*x) - n*m_y*m_x
    SS_xx = np.sum(x*x) - n*m_x*m_x
 
    # coeficientii regresiei 
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x
 
    return (b_0, b_1)
 
def plot_regression_line(x, y, b):
    # desenarea punctelor 
    plt.scatter(x, y, color = "m",
               marker = "o", s = 30)
 
    # vectorul de predictii
    y_pred = b[0] + b[1]*x
 
    # desenam linia 
    plt.plot(x, y_pred, color = "g")
 
    plt.xlabel('x')
    plt.ylabel('y')
 
    plt.show()
 
def main():
    #  data
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
 
    # estimating coefficients
    b = estimate_coef(x, y)
    print("Coeficientii estimati:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))
 
    # plotting regression line
    plot_regression_line(x, y, b)
 
if __name__ == "__main__":
    main()