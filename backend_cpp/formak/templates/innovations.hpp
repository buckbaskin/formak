if (_innovations.count(ReadingT::Identifier) > 0) {
  return std::any_cast<typename ReadingT::InnovationT>(
      _innovations[ReadingT::Identifier]);
}
return {};
