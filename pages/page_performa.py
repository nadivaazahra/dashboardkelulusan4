def model_page(df):
    st.title("🤖 Model Prediksi Kelulusan")
    df_processed = preprocess_data(df.copy())

    st.subheader("Pilih Rasio Data Latih")
    train_size = st.slider("Persentase Data untuk Training", 10, 90, 80, step=5)

    test_size = 1 - (train_size / 100)

    # Split data berdasarkan slider
    X = df_processed.drop('Status Kelulusan', axis=1)
    y = df_processed['Status Kelulusan']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    st.subheader("Akurasi Model")
    st.write(f"{accuracy:.2f}")

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    st.text(cr)
