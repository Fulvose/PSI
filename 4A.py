# przeskalowanie danych przez wartości własne
scaled_X = np.dot(X_std, e_vectors)

# narysowanie danych i wektorów własnych
plt.scatter(scaled_X[:,0], scaled_X[:,1], alpha=0.3)
plt.plot([0, 2*np.sqrt(e_values[1])*e_vectors[0,1]], [0, 2*np.sqrt(e_values[1])*e_vectors[1,1]], color='red')
plt.plot([0, 2*np.sqrt(e_values[0])*e_vectors[0,0]], [0, 2*np.sqrt(e_values[0])*e_vectors[1,0]], color='green')
plt.xlabel('Przestrzeń cech 1')
plt.ylabel('Przestrzeń cech 2')
plt.show()

